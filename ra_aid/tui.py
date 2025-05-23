from textual.app import App
from textual.widgets import Header, Footer, Input, Log, Static, ProgressBar, Tabs
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual import events

from ra_aid.config import load_config
from ra_aid.logging_config import get_logger
from ra_aid.database import connect_db

class RaAidTUI(App):
    """RA.Aid Terminal User Interface"""

    CSS = """
    .mode-indicator {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-align: center;
    }

    .agent-grid {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }

    .agent-card {
        border: solid $primary;
        height: 8;
        padding: 1;
    }

    .stats-panel {
        layout: vertical;
        height: 100%;
    }
    """

    BINDINGS = [
        ("c", "switch_mode('chat')", "Chat"),
        ("p", "switch_mode('parallel')", "Parallel"),
        ("s", "switch_mode('stats')", "Stats"),
        ("ctrl+o", "switch_mode('config')", "Config"),
        ("ctrl+e", "toggle_expert", "Expert Mode"),
        ("ctrl+a", "toggle_aider", "Aider"),
        ("ctrl+w", "toggle_web", "Web Research"),
        ("ctrl+c", "toggle_cowboy", "Cowboy Mode"),
        ("ctrl+t", "test_config", "Test Config"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+r", "restart_session", "Restart"),
        ("?", "help_screen", "Help")
    ]

    def __init__(self):
        super().__init__()
        # Initialize RA.Aid core components
        self.config = load_config()
        self.logger = get_logger()
        self.db = connect_db(".ra-aid/pk.db")

        # Mode management
        self.current_mode = "chat"
        self.mode_screens = {}

        # Session management
        self.current_session = None
        self.active_agents = {}

        # State tracking
        self.expert_mode = False
        self.aider_enabled = self.config.get('use_aider', True)
        self.cowboy_mode = self.config.get('cowboy_mode', False)
        self.web_research = self.config.get('web_research', True)

    def compose(self):
        """Create and yield widgets for the app."""
        with Container():
            yield Header(show_clock=True)
            yield Static(self._get_mode_indicator(), classes="mode-indicator", id="mode-indicator")
            yield Container(id="main-container")
            yield Footer()

    def on_mount(self):
        """Initialize the app when mounted."""
        self.switch_to_chat_mode()
        self.load_session()

    def _get_mode_indicator(self):
        """Get the mode indicator text."""
        indicators = []
        if self.expert_mode:
            indicators.append("🧠 Expert")
        if self.aider_enabled:
            indicators.append("🛠️ Aider")
        if self.cowboy_mode:
            indicators.append("🏇 Cowboy")
        if self.web_research:
            indicators.append("🌐 Web")

        mode_text = f"Mode: {self.current_mode.title()}"
        if indicators:
            mode_text += f" | {' | '.join(indicators)}"
        return mode_text

    def switch_to_chat_mode(self):
        """Switch to chat mode interface."""
        container = self.query_one("#main-container")
        container.remove_children()

        with container:
            with Vertical():
                yield Log(id="chat-output", wrap=True, auto_scroll=True)
                yield Input(
                    id="chat-input", 
                    placeholder="What would you like me to build today?",
                    name="chat-input"
                )

    def switch_to_parallel_mode(self):
        """Switch to parallel agents dashboard."""
        container = self.query_one("#main-container")
        container.remove_children()

        with container:
            with Vertical():
                yield Static("🔄 Parallel Agents Dashboard", id="parallel-header")
                with Container(classes="agent-grid", id="agent-grid"):
                    pass
                yield Input(
                    id="parallel-input",
                    placeholder="Delegate new task or manage agents...",
                    name="parallel-input"
                )

    def switch_to_stats_mode(self):
        """Switch to statistics view."""
        container = self.query_one("#main-container")
        container.remove_children()

        with container:
            with Vertical(classes="stats-panel"):
                yield Static("📊 RA.Aid Statistics", id="stats-header")
                yield Static(self._get_stats_content(), id="stats-content")

    def switch_to_config_mode(self):
        """Switch to configuration mode."""
        container = self.query_one("#main-container")
        container.remove_children()

        with container:
            with Vertical():
                yield Static("⚙️ Configuration", id="config-header")
                yield Static(self._get_config_content(), id="config-content")

    async def on_input_submitted(self, message: Input.Submitted):
        """Handle input submission based on current mode."""
        input_widget = message.input
        input_text = message.value

        # Clear input
        input_widget.value = ""

        if self.current_mode == "chat":
            await self._handle_chat_input(input_text)
        elif self.current_mode == "parallel":
            await self._handle_parallel_input(input_text)

    async def _handle_chat_input(self, input_text: str):
        """Handle chat mode input."""
        output = self.query_one("#chat-output")
        output.write(f"\n> {input_text}")

        # Show processing indicator
        output.write("🤔 Thinking...")

        try:
            # Process through RA.Aid's three-stage architecture
            result = await self._process_ra_aid_task(input_text)
            output.write(f"✅ {result}")
        except Exception as e:
            output.write(f"❌ Error: {str(e)}")

    async def _handle_parallel_input(self, input_text: str):
        """Handle parallel mode input."""
        # Parse input for agent commands
        if input_text.startswith("/spawn"):
            agent_type = input_text.split()[1] if len(input_text.split()) > 1 else "general"
            task = " ".join(input_text.split()[2:])
            await self._spawn_agent(agent_type, task)
        else:
            # Default: create general agent
            await self._spawn_agent("general", input_text)

    async def _process_ra_aid_task(self, task: str):
        """Process task through RA.Aid's architecture."""
        # This would integrate with existing RA.Aid components
        # For now, placeholder
        return f"Processed: {task}"

    async def _spawn_agent(self, agent_type: str, task: str):
        """Spawn a new agent for parallel processing."""
        agent_id = f"{agent_type}_{len(self.active_agents)}"

        # Create agent card
        grid = self.query_one("#agent-grid")
        agent_card = Static(
            f"🤖 {agent_type.title()}\n"
            f"Task: {task[:30]}...\n"
            f"Status: Starting...\n"
            f"Progress: 0%",
            classes="agent-card",
            id=f"agent-{agent_id}"
        )
        grid.mount(agent_card)

        # Store agent reference
        self.active_agents[agent_id] = {
            'type': agent_type,
            'task': task,
            'status': 'starting',
            'progress': 0
        }

    def _get_stats_content(self):
        """Generate statistics content."""
        return (
            "📈 Today's Activity:\n"
            "  • Tasks Completed: 12\n"
            "  • Success Rate: 94%\n"
            "  • Avg Time: 4.2 min\n"
            "  • Cost: $2.34\n\n"
            "🛠️ Tool Usage:\n"
            "  • Aider: 8 tasks\n"
            "  • Expert: 3 tasks\n"
            "  • Web Research: 2 tasks\n\n"
            "🎯 Top Performing:\n"
            "  • Backend tasks: 3.1 min avg\n"
            "  • Frontend tasks: 6.8 min avg"
        )

    def _get_config_content(self):
        """Generate configuration content."""
        return (
            f"🤖 Models:\n"
            f"  Primary: {self.config.get('model', 'claude-3.5-sonnet')}\n"
            f"  Expert: {self.config.get('expert_model', 'o1')}\n"
            f"  Fallback: {self.config.get('fallback_model', 'gpt-4o')}\n\n"
            f"⚙️ Settings:\n"
            f"  Expert Mode: {'✓' if self.expert_mode else '✗'}\n"
            f"  Aider Integration: {'✓' if self.aider_enabled else '✗'}\n"
            f"  Cowboy Mode: {'✓' if self.cowboy_mode else '✗'}\n"
            f"  Web Research: {'✓' if self.web_research else '✗'}\n\n"
            f"💰 Limits:\n"
            f"  Daily Cost Limit: ${self.config.get('cost_limit', 10)}\n"
            f"  Current Usage: $2.34"
        )

    async def action_switch_mode(self, mode: str):
        """Switch between different modes."""
        self.current_mode = mode

        if mode == "chat":
            self.switch_to_chat_mode()
        elif mode == "parallel":
            self.switch_to_parallel_mode()
        elif mode == "stats":
            self.switch_to_stats_mode()
        elif mode == "config":
            self.switch_to_config_mode()

        # Update mode indicator
        self.query_one("#mode-indicator").update(self._get_mode_indicator())
        self.notify(f"Switched to {mode} mode")

    async def action_toggle_expert(self):
        """Toggle expert mode."""
        self.expert_mode = not self.expert_mode
        self.config['expert_mode'] = self.expert_mode
        self.query_one("#mode-indicator").update(self._get_mode_indicator())
        self.notify(f"Expert mode: {'ON' if self.expert_mode else 'OFF'}")

    async def action_toggle_aider(self):
        """Toggle aider integration."""
        self.aider_enabled = not self.aider_enabled
        self.config['use_aider'] = self.aider_enabled
        self.query_one("#mode-indicator").update(self._get_mode_indicator())
        self.notify(f"Aider: {'ENABLED' if self.aider_enabled else 'DISABLED'}")

    async def action_toggle_cowboy(self):
        """Toggle cowboy mode."""
        self.cowboy_mode = not self.cowboy_mode
        self.config['cowboy_mode'] = self.cowboy_mode
        self.query_one("#mode-indicator").update(self._get_mode_indicator())
        self.notify(f"Cowboy mode: {'ON' if self.cowboy_mode else 'OFF'}")

    async def action_toggle_web(self):
        """Toggle web research."""
        self.web_research = not self.web_research
        self.config['web_research'] = self.web_research
        self.query_one("#mode-indicator").update(self._get_mode_indicator())
        self.notify(f"Web research: {'ENABLED' if self.web_research else 'DISABLED'}")

    def load_session(self):
        """Load previous session from database."""
        # Integrate with RA.Aid's existing session management
        pass

    async def action_restart_session(self):
        """Restart the current session."""
        self.notify("Session restarted")
        if self.current_mode == "chat":
            self.query_one("#chat-output").clear()

def main():
    """Entry point for the TUI."""
    app = RaAidTUI()
    app.run()

if __name__ == "__main__":
    main()