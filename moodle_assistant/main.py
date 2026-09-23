"""
MoodleAssist-AI - Interfaz de Línea de Comandos (CLI)
Consola interactiva con Rich: visualización de preguntas, razonamiento de la IA,
modos asistido, autónomo y dry-run, pausas de confirmación y reportes de sesión.
"""

from __future__ import annotations

import os
import sys
import asyncio
from datetime import datetime
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import box

from .config import settings, Settings
from .models import (
    QuestionData,
    QuestionType,
    LLMResponse,
    LogEntry,
    QuizSessionReport,
    ActionExecutionResult,
)
from .dom_parser import DOMParser
from .llm_engine import LLMEngine
from .automator import MoodleAutomator

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False)


def print_banner() -> None:
    """Imprime el banner inicial con estilo Rich."""
    title = Text("🚀 MoodleAssist-AI v1.0", style="bold magenta")
    subtitle = Text(
        "Automatización Inteligente para Cuestionarios y Actividades en Moodle\n"
        f"Motor: Groq ({settings.groq_model}) | CDP: nodriver",
        style="dim white",
    )
    banner_content = Text.assemble(title, "\n", subtitle)
    console.print(
        Panel(
            banner_content,
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def print_config_status(cfg: Settings) -> None:
    """Muestra el estado de la configuración actual."""
    table = Table(title="🔧 Configuración del Sistema", box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("Parámetro", style="cyan", no_wrap=True)
    table.add_column("Valor Actual", style="green")
    table.add_column("Estado", style="bold")

    # API Key
    if cfg.is_groq_configured():
        masked_key = f"{cfg.groq_api_key[:6]}...{cfg.groq_api_key[-4:]}"
        table.add_row("GROQ_API_KEY", masked_key, "[green]Configurada[/green]")
    else:
        table.add_row("GROQ_API_KEY", "No establecida o inválida", "[red]Falta en .env[/red]")

    table.add_row("Modelo Groq", cfg.groq_model, "[cyan]Activo[/cyan]")
    table.add_row("Modo Headless", str(cfg.headless), "[dim]Ventana visible[/dim]" if not cfg.headless else "[yellow]Sin ventana[/yellow]")
    table.add_row("Perfil Chrome", str(cfg.resolve_user_data_dir()), "[dim]Persistente[/dim]")
    table.add_row("Delay Humano", f"{cfg.human_delay_min}s - {cfg.human_delay_max}s", "[dim]Activo[/dim]")

    console.print(table)
    console.print()


def find_matching_answer(q: QuestionData, q_idx: int, answers: List[LLMResponse], total_questions: int) -> Optional[LLMResponse]:
    """Busca de forma tolerante la respuesta correspondiente a una pregunta."""
    if not answers:
        return None
    # 1. Búsqueda exacta por question_id
    for ans in answers:
        if ans.question_id == q.question_id:
            return ans
    # 2. Búsqueda por número visible o id normalizado
    q_num_clean = q.number.strip().lower()
    for ans in answers:
        ans_id_clean = ans.question_id.strip().lower()
        if ans_id_clean == q_num_clean or ans_id_clean in q_num_clean or q_num_clean in ans_id_clean:
            return ans
    # 3. Búsqueda por dígitos
    q_digits = "".join(filter(str.isdigit, q.number))
    if q_digits:
        for ans in answers:
            ans_digits = "".join(filter(str.isdigit, ans.question_id))
            if ans_digits and ans_digits == q_digits:
                return ans
    # 4. Fallback posicional si las listas tienen el mismo tamaño
    if len(answers) == total_questions and q_idx < len(answers):
        return answers[q_idx]
    # 5. Fallback si solo hay 1 elemento en ambos
    if len(answers) == 1 and total_questions == 1:
        return answers[0]
    return None


def render_questions_table(questions: List[QuestionData], answers: List[LLMResponse]) -> None:
    """Renderiza una tabla detallada con las preguntas encontradas y las respuestas propuestas por la IA."""
    table = Table(
        title="📝 Respuestas Propuestas por MoodleAssist-AI",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("# / ID", style="bold yellow", width=12)
    table.add_column("Tipo", style="magenta", width=14)
    table.add_column("Enunciado", style="white", min_width=25, max_width=45)
    table.add_column("Respuesta Sugerida", style="bold green", min_width=25, max_width=45)
    table.add_column("Razonamiento (IA)", style="italic dim", min_width=20, max_width=35)
    table.add_column("Conf.", style="bold cyan", justify="right", width=8)

    total_q = len(questions)
    for q_idx, q in enumerate(questions):
        ans = find_matching_answer(q, q_idx, answers, total_q)
        
        # Formatear respuesta sugerida
        suggested_text = "No determinada"
        reasoning = "N/A"
        confidence_str = "0%"

        if ans:
            confidence_str = f"{int(ans.confidence * 100)}%"
            reasoning = ans.explanation or "Sin justificación"

            if q.qtype in (QuestionType.MULTICHOICE, QuestionType.TRUEFALSE):
                if ans.selected_options:
                    suggested_text = "\n".join([f"• {opt}" for opt in ans.selected_options])
                else:
                    suggested_text = "[yellow]Ninguna marcada[/yellow]"

            elif q.qtype in (QuestionType.SHORTANSWER, QuestionType.NUMERICAL, QuestionType.ESSAY):
                suggested_text = f"\"{ans.text_answer or ''}\""

            elif q.qtype in (QuestionType.MATCH, QuestionType.GAPSELECT, QuestionType.SELECTMULTI, QuestionType.MULTIANSWER):
                if ans.matching_answers:
                    pairs = []
                    for sq in q.sub_questions:
                        val = ans.matching_answers.get(sq.sub_id)
                        if val is None:
                            for k, v in ans.matching_answers.items():
                                if k.lower() == sq.sub_id.lower() or k.lower() in sq.sub_id.lower() or sq.sub_id.lower() in k.lower():
                                    val = v
                                    break
                        if val is not None:
                            opt_label = str(val)
                            for opt in sq.options:
                                if opt.value == str(val) or opt.id == str(val) or opt.text.lower() == str(val).lower():
                                    opt_label = opt.text
                                    break
                            sq_short = sq.sub_id
                            if "_sub" in sq_short:
                                sq_short = sq_short.split("_sub")[-1].replace("_answer", "")
                                sq_short = f"#{sq_short}"
                            pairs.append(f"• {sq_short}: {opt_label}")
                    
                    if not pairs:
                        pairs = [f"• {k}: {v}" for k, v in list(ans.matching_answers.items())]

                    if len(pairs) > 6:
                        suggested_text = "\n".join(pairs[:6]) + f"\n... (+{len(pairs)-6} más)"
                    else:
                        suggested_text = "\n".join(pairs)
                elif ans.text_answer:
                    suggested_text = f"\"{ans.text_answer}\""
                else:
                    suggested_text = "[yellow]Sin respuestas asignadas[/yellow]"

        # Acortar prompt si es muy largo
        clean_p = q.prompt.replace("\n", " ")
        if len(clean_p) > 120:
            clean_p = clean_p[:117] + "..."

        table.add_row(
            f"{q.number}\n({q.question_id})",
            q.qtype.value,
            clean_p,
            suggested_text,
            reasoning,
            confidence_str,
        )

    console.print(table)


class MoodleAssistCLI:
    """Controlador principal del CLI interactivo."""

    def __init__(self, cfg: Settings = settings):
        self.config = cfg
        self.llm = LLMEngine(cfg)
        self.automator = MoodleAutomator(cfg)
        self.report = QuizSessionReport()

    async def initialize(self) -> None:
        """Verifica la configuración inicial e interactúa con el usuario si faltan datos."""
        print_banner()
        print_config_status(self.config)

        if not self.config.is_groq_configured():
            console.print(
                "[bold yellow]⚠️ No se detectó una GROQ_API_KEY válida.[/bold yellow]\n"
                "Para obtener una clave gratuita dirígete a: [link=https://console.groq.com/keys]https://console.groq.com/keys[/link]\n"
            )
            key = Prompt.ask("[bold green]Ingresa tu GROQ_API_KEY ahora (o presiona Enter para configurar en .env luego)[/bold green]")
            if key and not key.startswith("tu_"):
                self.config.groq_api_key = key.strip()
                # Guardar en .env
                env_path = self.config.resolve_user_data_dir().parent / ".env"
                try:
                    with open(env_path, "a", encoding="utf-8") as f:
                        f.write(f"\nGROQ_API_KEY={self.config.groq_api_key}\n")
                    console.print(f"[green]✓ Clave guardada en {env_path}[/green]")
                except Exception:
                    pass

    async def run_menu(self) -> None:
        """Bucle principal del menú interactivo."""
        await self.initialize()

        while True:
            console.print("\n[bold cyan]═══ MENÚ PRINCIPAL ═══[/bold cyan]")
            console.print("[1] 🌐 [bold]Iniciar Navegador y Login en Moodle[/bold]")
            console.print("[2] 🎯 [bold]Resolver Página Actual (Modo Asistido / Paso a Paso)[/bold]")
            console.print("[3] ⚡ [bold]Resolver Cuestionario Completo (Modo Autónomo / Full Auto)[/bold]")
            console.print("[4] 🔍 [bold]Modo Dry-Run (Analizar y Consultar Respuestas SIN interactuar)[/bold]")
            console.print("[5] 📸 [bold]Capturar Pantalla de la Página Actual[/bold]")
            console.print("[6] 🧪 [bold]Probar Conexión con Groq API[/bold]")
            console.print("[0] 🚪 [bold red]Salir[/bold red]")

            choice = Prompt.ask("\nSelecciona una opción", choices=["0", "1", "2", "3", "4", "5", "6"], default="2")

            if choice == "1":
                await self.action_open_browser_and_login()
            elif choice == "2":
                await self.action_solve_current_page(dry_run=False, ask_confirmation=True)
            elif choice == "3":
                await self.action_solve_full_quiz()
            elif choice == "4":
                await self.action_solve_current_page(dry_run=True, ask_confirmation=False)
            elif choice == "5":
                await self.action_screenshot()
            elif choice == "6":
                await self.action_test_groq()
            elif choice == "0":
                await self.action_exit()
                break

    async def ensure_browser_running(self) -> bool:
        """Asegura que el navegador esté iniciado."""
        if self.automator.browser is None:
            with console.status("[bold cyan]Iniciando navegador nodriver con perfil persistente..."):
                try:
                    await self.automator.start()
                    console.print("[bold green]✓ Navegador iniciado correctamente.[/bold green]")
                except Exception as e:
                    console.print(f"[bold red]✗ Error al iniciar navegador: {e}[/bold red]")
                    return False

            if self.config.moodle_url:
                await self.automator.navigate(self.config.moodle_url)

        return True

    async def action_open_browser_and_login(self) -> None:
        """Abre el navegador para que el usuario inicie sesión y navegue al cuestionario."""
        ok = await self.ensure_browser_running()
        if not ok:
            return

        target_url = self.config.moodle_url or Prompt.ask(
            "Ingresa la URL de tu campus Moodle o cuestionario (o Enter para dejar la pestaña actual)"
        )
        if target_url:
            await self.automator.navigate(target_url)

        console.print(
            Panel(
                "[bold green]Navegador Listo[/bold green]\n\n"
                "1. En la ventana de Chrome que se abrió, inicia sesión en tu Moodle con tus credenciales.\n"
                "2. Navega hasta el cuestionario, taller o actividad que deseas resolver.\n"
                "3. Inicia o continúa el intento del cuestionario.\n"
                "4. Cuando estés viendo las preguntas, regresa aquí y selecciona la opción [bold cyan]2[/bold cyan] o [bold cyan]3[/bold cyan].",
                title="Paso de Autenticación y Navegación",
                border_style="green",
            )
        )
        Prompt.ask("\nPresiona [bold]Enter[/bold] cuando estés en la página del cuestionario")

    async def action_solve_current_page(self, dry_run: bool = False, ask_confirmation: bool = True) -> int:
        """Extrae, resuelve y opcionalmente aplica las respuestas en la página actual."""
        ok = await self.ensure_browser_running()
        if not ok:
            return 0

        # Verificar si hay preguntas
        with console.status("[bold yellow]Extrayendo elementos del DOM con CDP..."):
            questions = await self.automator.extract_questions()

        if not questions:
            console.print(
                "[bold yellow]⚠️ No se detectaron preguntas (elementos 'div.que') en la página actual.[/bold yellow]\n"
                "Asegúrate de que estás en la pantalla activa de intento del cuestionario (ej: /mod/quiz/attempt.php)."
            )
            return 0

        console.print(f"[bold green]✓ Se detectaron {len(questions)} pregunta(s) en la página.[/bold green]")

        # Consultar al LLM
        with console.status(f"[bold cyan]Analizando y resolviendo con Groq ({self.config.groq_model})..."):
            try:
                answers = await self.llm.solve_batch_questions(questions)
            except Exception as e:
                console.print(f"[bold red]✗ Error al comunicarse con Groq: {e}[/bold red]")
                return 0

        # Mostrar tabla de resultados
        render_questions_table(questions, answers)

        if dry_run:
            console.print("[bold yellow]ℹ Modo Dry-Run activo: No se realizaron cambios en la página.[/bold yellow]")
            return len(questions)

        # Confirmación interactiva si aplica
        if ask_confirmation:
            should_apply = Confirm.ask(
                "\n[bold green]¿Deseas aplicar estas respuestas automáticamente en el navegador?[/bold green]",
                default=True
            )
            if not should_apply:
                console.print("[yellow]Operación cancelada por el usuario.[/yellow]")
                return len(questions)

        # Aplicar respuestas en el DOM
        console.print("\n[bold cyan]Aplicando respuestas en el navegador...[/bold cyan]")
        success_count = 0
        total_q = len(questions)

        for q_idx, q in enumerate(questions):
            ans = find_matching_answer(q, q_idx, answers, total_q)
            if not ans:
                console.print(f"[yellow]⚠️ No se halló solución generada para {q.number} ({q.question_id})[/yellow]")
                continue

            with console.status(f"Completando {q.number}..."):
                res: ActionExecutionResult = await self.automator.apply_solution(q, ans)

            status_style = "green" if res.success else "red"
            mark = "✓" if res.success else "✗"
            console.print(f"[{status_style}]  {mark} {q.number}: {res.action_taken}[/{status_style}]")

            if res.success:
                success_count += 1

            # Registrar en reporte
            self.report.entries.append(
                LogEntry(
                    question_id=q.question_id,
                    question_number=q.number,
                    question_type=q.qtype,
                    chosen_action=res.action_taken,
                    confidence=ans.confidence,
                    status="SUCCESS" if res.success else "ERROR",
                    details=res.error_message,
                )
            )

        self.report.total_questions += len(questions)
        self.report.solved_questions += success_count

        # Sincronizar autosave nativo de Moodle
        with console.status("Sincronizando cambios con autosave de Moodle..."):
            await self.automator.trigger_moodle_autosave()

        console.print(f"\n[bold green]✓ {success_count}/{len(questions)} respuestas aplicadas exitosamente.[/bold green]")

        if ask_confirmation:
            console.print(
                Panel(
                    "[bold yellow]⚠️ Importante para guardar en Moodle:[/bold yellow]\n\n"
                    "En los cuestionarios de Moodle, las respuestas se registran de forma definitiva en el servidor\n"
                    "cuando se presiona [bold cyan]'Siguiente página'[/bold cyan] o [bold cyan]'Terminar intento ...'[/bold cyan].\n"
                    "Si sales o recargas la página sin enviarla, el servidor de Moodle no habrá guardado las respuestas.",
                    title="Persistencia de Respuestas",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
            should_save_nav = Confirm.ask(
                "[bold green]¿Deseas presionar 'Siguiente página / Terminar intento' para guardar en el servidor ahora?[/bold green]",
                default=True
            )
            if should_save_nav:
                with console.status("[bold cyan]Guardando y avanzando página en Moodle..."):
                    clicked = await self.automator.click_next_page()
                if clicked:
                    console.print("[bold green]✓ Página enviada y guardada en el servidor de Moodle.[/bold green]")
                else:
                    console.print("[bold yellow]No se detectó botón de avance automático. Puedes hacer clic en 'Finish attempt ...' en Chrome.[/bold yellow]")

        return len(questions)

    async def action_solve_full_quiz(self) -> None:
        """Modo autónomo: resuelve página a página avanzando automáticamente."""
        ok = await self.ensure_browser_running()
        if not ok:
            return

        page_number = 1
        console.print(
            Panel(
                "[bold yellow]Iniciando Modo Autónomo (Full Auto)[/bold yellow]\n\n"
                "El asistente resolverá la página actual y avanzará a la siguiente de manera continua.\n"
                "Por seguridad, puedes pausar entre páginas o dejar que avance automáticamente.",
                border_style="yellow",
            )
        )
        auto_advance = Confirm.ask("¿Deseas avanzar automáticamente entre páginas sin preguntar?", default=True)

        while True:
            console.print(f"\n[bold cyan]═══════════════ PROCESANDO PÁGINA {page_number} ═══════════════[/bold cyan]")
            q_count = await self.action_solve_current_page(dry_run=False, ask_confirmation=False)

            if q_count == 0:
                console.print("[yellow]No hay más preguntas en esta página o se ha finalizado el intento.[/yellow]")
                break

            # Intentar avanzar a la siguiente página
            if not auto_advance:
                cont = Confirm.ask("¿Continuar a la siguiente página?", default=True)
                if not cont:
                    break

            with console.status("[bold cyan]Avanzando a la siguiente página..."):
                clicked = await self.automator.click_next_page()

            if not clicked:
                console.print("[bold yellow]No se encontró botón para avanzar. El cuestionario puede haber terminado.[/bold yellow]")
                break

            page_number += 1
            await asyncio.sleep(1.0)

        console.print("\n[bold green]🏁 Sesión autónoma completada.[/bold green]")

    async def action_screenshot(self) -> None:
        """Captura un pantallazo de la pestaña activa."""
        ok = await self.ensure_browser_running()
        if not ok:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/moodle_quiz_{ts}.png"
        path = await self.automator.take_screenshot(filename)
        if path:
            console.print(f"[bold green]✓ Pantallazo guardado en: {path}[/bold green]")
        else:
            console.print("[bold red]✗ No se pudo capturar la pantalla.[/bold red]")

    async def action_test_groq(self) -> None:
        """Prueba de conectividad con la API de Groq."""
        with console.status(f"[bold cyan]Probando conexión con Groq ({self.config.groq_model})..."):
            ok = await self.llm.test_connection()
        if ok:
            console.print("[bold green]✓ Conexión exitosa con Groq API y modelo funcional.[/bold green]")
        else:
            console.print(
                "[bold red]✗ Falló la conexión con Groq API.[/bold red]\n"
                "Verifica que tu GROQ_API_KEY sea correcta y que tengas conexión a Internet."
            )

    async def action_exit(self) -> None:
        """Cierra el navegador y muestra el reporte final."""
        console.print("\n[bold cyan]Cerrando sesión de MoodleAssist-AI...[/bold cyan]")
        await self.automator.stop()
        self.report.finished_at = datetime.now()

        # Mostrar resumen si se respondieron preguntas
        if self.report.total_questions > 0:
            rep_table = Table(title="📊 Reporte de Sesión", box=box.ROUNDED)
            rep_table.add_column("Métrica", style="cyan")
            rep_table.add_column("Resultado", style="bold green")

            rep_table.add_row("Total Preguntas Procesadas", str(self.report.total_questions))
            rep_table.add_row("Preguntas Aplicadas con Éxito", str(self.report.solved_questions))
            rep_table.add_row("Fallos de Interacción", str(self.report.total_questions - self.report.solved_questions))
            console.print(rep_table)

        console.print("[bold magenta]¡Hasta pronto![/bold magenta]\n")


def main() -> None:
    """Punto de entrada síncrono para el CLI."""
    cli = MoodleAssistCLI()
    try:
        asyncio.run(cli.run_menu())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupción por teclado (Ctrl+C). Saliendo limpiamente...[/bold yellow]")
        if cli.automator.browser:
            asyncio.run(cli.automator.stop())
        sys.exit(0)


if __name__ == "__main__":
    main()

