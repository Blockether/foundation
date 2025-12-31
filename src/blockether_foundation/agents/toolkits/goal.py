"""
Goal management toolkit for interactive/dynamic goal management during agent sessions.

Provides tools for creating, managing, and tracking goals with hierarchical
structure, priority levels, and status lifecycle. Goals are stored in session
state and persist throughout the agent conversation.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from agno.run import RunContext
from agno.tools import Toolkit

from blockether_foundation.models import (
    Goal,
    GoalPriority,
    GoalStatus,
)

type GoalsDict = dict[str, Goal]


def _ensure_session_state(run_context: RunContext) -> tuple[GoalsDict, str | None]:
    """Ensure session state is initialized and return goals dict and current goal ID."""
    if run_context.session_state is None:
        run_context.session_state = {"goals": {}, "current_goal_id": None}

    if "goals" not in run_context.session_state:
        run_context.session_state["goals"] = {}

    if "current_goal_id" not in run_context.session_state:
        run_context.session_state["current_goal_id"] = None

    goals_data = run_context.session_state.get("goals", {})
    # Cast to GoalsDict - at runtime we trust that the session_state contains the correct types
    goals: GoalsDict = cast(GoalsDict, goals_data) if isinstance(goals_data, dict) else {}

    current_goal_id_val = run_context.session_state.get("current_goal_id")
    current_goal_id: str | None = (
        current_goal_id_val if isinstance(current_goal_id_val, str) else None
    )

    return goals, current_goal_id


class GoalToolkit(Toolkit):
    """Toolkit for goal management with hierarchical structure and lifecycle tracking."""

    def __init__(self, **kwargs: Any):
        tools: list[Callable[..., str]] = [
            self.create_goal,
            self.list_goals,
            self.finish_goal,
            self.skip_goal,
            self.refine_goal,
            self.jump_to_goal,
            self.set_goal_priority,
            self.get_current_goal,
        ]

        instructions = """
        Use these tools for interactive/dynamic goal management during the agent session.
        Goals persist in session state throughout the conversation.

        Lifecycle:
        - create_goal: Create new goals with priority, importance, and subgoals
        - list_goals: View all goals with status
        - finish_goal: Mark goals as completed
        - skip_goal: Skip goals without finishing
        - refine_goal: Update goal attributes
        - jump_to_goal: Switch active goal
        - set_goal_priority: Change goal priority
        - get_current_goal: View current goal

        Priority levels: P0 (critical), P1 (high), P2 (medium), P3 (low)
        Status states: pending, in_progress, completed, skipped, refined
        Importance: 0.0-1.0
        """

        super().__init__(
            name="goal_management",
            tools=tools,
            instructions=instructions,
            add_instructions=True,
            **kwargs,
        )

    def create_goal(
        self,
        run_context: RunContext,
        title: str,
        description: str,
        importance: float = 0.5,
        priority: str = "P2",
        parent_goal_id: str | None = None,
        summary: str | None = None,
    ) -> str:
        """Create a new goal with title, description, importance, priority, and optional subgoals."""
        goals, current_goal_id = _ensure_session_state(run_context)
        assert run_context.session_state is not None

        goal_id = f"goal_{len(goals) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        try:
            priority_enum = GoalPriority[priority]
        except KeyError:
            priority_enum = GoalPriority.MEDIUM

        goal = Goal(
            id=goal_id,
            title=title,
            description=description,
            summary=summary,
            priority=priority_enum,
            importance=importance,
            status=GoalStatus.PENDING,
            subgoals=[],
            parent_goal_id=parent_goal_id,
        )

        goals[goal_id] = goal

        if parent_goal_id and parent_goal_id in goals:
            parent_goal = goals[parent_goal_id]
            parent_goal.subgoals.append(goal)

        if not current_goal_id:
            run_context.session_state["current_goal_id"] = goal_id

        return (
            f"Created goal '{title}' (ID: {goal_id})\n"
            f"  Priority: {priority}\n"
            f"  Importance: {importance}\n"
            f"  Status: pending"
        )

    def list_goals(
        self,
        run_context: RunContext,
        include_finished: bool = True,
        include_subgoals: bool = False,
    ) -> str:
        """List all goals with their status, priority, and details."""
        goals, current_goal_id = _ensure_session_state(run_context)

        if not goals:
            return "No goals exist yet. Use create_goal to create your first goal."

        result = ["Goals:", ""]

        for goal_id, goal in goals.items():
            status_marker = "← CURRENT" if goal_id == current_goal_id else ""
            status_display = f"[{goal.status.value.upper()}]"

            result.append(f"{status_display} {goal.title} (ID: {goal_id}) {status_marker}")
            result.append(f"  Priority: {goal.priority.name} | Importance: {goal.importance}")

            if goal.summary:
                result.append(f"  Summary: {goal.summary}")

            if goal.description:
                desc_lines = goal.description.split("\n")
                result.append(f"  Description: {desc_lines[0]}")
                if len(desc_lines) > 1:
                    result.append(f"    ({len(desc_lines) - 1} more lines)")

            if include_subgoals and goal.subgoals:
                result.append(f"  Subgoals ({len(goal.subgoals)}):")
                for i, subgoal in enumerate(goal.subgoals, 1):
                    result.append(f"    {i}. {subgoal.title} [{subgoal.status.value}]")

            result.append("")

        return "\n".join(result)

    def finish_goal(
        self,
        run_context: RunContext,
        goal_id: str | None = None,
        summary: str | None = None,
    ) -> str:
        """Finish a goal and mark it as completed."""
        goals, current_goal_id = _ensure_session_state(run_context)
        assert run_context.session_state is not None
        target_goal_id = goal_id or current_goal_id

        if not target_goal_id or target_goal_id not in goals:
            return "No valid goal to finish. Use list_goals to see available goals."

        goal = goals[target_goal_id]
        goal.status = GoalStatus.COMPLETED
        goal.updated_at = datetime.now().isoformat()

        if summary:
            goal.summary = summary

        next_goal_id = self._find_next_goal(goals)
        if next_goal_id:
            run_context.session_state["current_goal_id"] = next_goal_id
            goals[next_goal_id].status = GoalStatus.IN_PROGRESS
            return (
                f"Completed goal: {goal.title}\n"
                f"  Summary: {summary or goal.summary or 'No summary provided'}\n"
                f"\n"
                f"Next goal: {goals[next_goal_id].title}"
            )
        else:
            run_context.session_state["current_goal_id"] = None
            return (
                f"Completed goal: {goal.title}\n"
                f"  Summary: {summary or goal.summary or 'No summary provided'}\n"
                f"\n"
                f"All goals completed."
            )

    def skip_goal(
        self,
        run_context: RunContext,
        goal_id: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Skip a goal without marking it as finished."""
        goals, current_goal_id = _ensure_session_state(run_context)
        assert run_context.session_state is not None
        target_goal_id = goal_id or current_goal_id

        if not target_goal_id or target_goal_id not in goals:
            return "No valid goal to skip. Use list_goals to see available goals."

        goal = goals[target_goal_id]
        goal.status = GoalStatus.SKIPPED
        goal.updated_at = datetime.now().isoformat()

        next_goal_id = self._find_next_goal(goals)
        if next_goal_id:
            run_context.session_state["current_goal_id"] = next_goal_id
            goals[next_goal_id].status = GoalStatus.IN_PROGRESS
            return (
                f"Skipped goal: {goal.title}\n"
                f"  Reason: {reason or 'Not specified'}\n"
                f"\n"
                f"Next goal: {goals[next_goal_id].title}"
            )
        else:
            run_context.session_state["current_goal_id"] = None
            return (
                f"Skipped goal: {goal.title}\n"
                f"  Reason: {reason or 'Not specified'}\n"
                f"\n"
                f"No more pending goals."
            )

    def refine_goal(
        self,
        run_context: RunContext,
        goal_id: str | None = None,
        new_title: str | None = None,
        new_description: str | None = None,
        new_importance: float | None = None,
        new_priority: str | None = None,
    ) -> str:
        """Refine or update an existing goal's attributes."""
        goals, current_goal_id = _ensure_session_state(run_context)
        target_goal_id = goal_id or current_goal_id

        if not target_goal_id or target_goal_id not in goals:
            return "No valid goal to refine. Use list_goals to see available goals."

        goal = goals[target_goal_id]

        changes: list[str] = []

        if new_title:
            goal.title = new_title
            changes.append(f"title -> {new_title}")

        if new_description:
            goal.description = new_description
            changes.append("description updated")

        if new_importance is not None:
            goal.importance = max(0.0, min(1.0, new_importance))
            changes.append(f"importance -> {goal.importance}")

        if new_priority:
            priority_map = {
                "P0": GoalPriority.HIGHEST,
                "P1": GoalPriority.HIGH,
                "P2": GoalPriority.MEDIUM,
                "P3": GoalPriority.LOW,
            }
            if new_priority in priority_map:
                goal.priority = priority_map[new_priority]
                changes.append(f"priority -> {new_priority}")

        if changes:
            goal.status = GoalStatus.REFINED
            goal.updated_at = datetime.now().isoformat()
            return f"Refined goal '{goal.title}':\n  " + ", ".join(changes)
        else:
            return "No changes specified to refine goal."

    def jump_to_goal(
        self,
        run_context: RunContext,
        goal_id: str,
    ) -> str:
        """Switch to a specific goal and mark it as current goal."""
        goals, _ = _ensure_session_state(run_context)
        assert run_context.session_state is not None

        if goal_id not in goals:
            return f"Goal '{goal_id}' not found. Use list_goals to see available goals."

        goal = goals[goal_id]

        if goal.status == GoalStatus.COMPLETED:
            return f"Goal '{goal.title}' is already completed. Cannot jump to it."

        run_context.session_state["current_goal_id"] = goal_id
        goal.status = GoalStatus.IN_PROGRESS
        goal.updated_at = datetime.now().isoformat()

        return (
            f"Switched to goal: {goal.title}\n"
            f"  Priority: {goal.priority.name} | Importance: {goal.importance}\n"
            f"  Status: {goal.status.value}"
        )

    def set_goal_priority(
        self,
        run_context: RunContext,
        goal_id: str | None = None,
        priority: str = "P2",
    ) -> str:
        """Change the priority of a goal."""
        goals, current_goal_id = _ensure_session_state(run_context)
        target_goal_id = goal_id or current_goal_id

        if not target_goal_id or target_goal_id not in goals:
            return "No valid goal to update. Use list_goals to see available goals."

        goal = goals[target_goal_id]

        priority_map = {
            "P0": GoalPriority.HIGHEST,
            "P1": GoalPriority.HIGH,
            "P2": GoalPriority.MEDIUM,
            "P3": GoalPriority.LOW,
        }
        if priority in priority_map:
            old_priority = goal.priority.name
            goal.priority = priority_map[priority]
            goal.updated_at = datetime.now().isoformat()
            return f"Updated goal '{goal.title}': priority {old_priority} -> {priority}"
        else:
            return f"Invalid priority level '{priority}'. Use P0, P1, P2, or P3."

    def get_current_goal(
        self,
        run_context: RunContext,
    ) -> str:
        """Get the current active goal with full details."""
        goals, current_goal_id = _ensure_session_state(run_context)

        if not current_goal_id or not goals or current_goal_id not in goals:
            return "No active goal set. Use create_goal to create a goal or jump_to_goal to select one."

        goal = goals[current_goal_id]

        result = [
            f"Current goal: {goal.title}",
            f"ID: {goal.id}",
            f"Priority: {goal.priority.name}",
            f"Importance: {goal.importance}",
            f"Status: {goal.status.value}",
        ]

        if goal.summary:
            result.append(f"Summary: {goal.summary}")

        result.append(f"\nDescription:\n{goal.description}")

        if goal.subgoals:
            result.append(f"\nSubgoals ({len(goal.subgoals)}):")
            for i, subgoal in enumerate(goal.subgoals, 1):
                result.append(f"  {i}. {subgoal.title} [{subgoal.status.value}]")

        if goal.parent_goal_id:
            result.append(f"\nParent goal: {goal.parent_goal_id}")

        return "\n".join(result)

    def _find_next_goal(self, goals: GoalsDict, exclude_goal_id: str | None = None) -> str | None:
        """Find the next pending goal by priority, excluding specified goal ID."""
        pending_goals = [
            (gid, g)
            for gid, g in goals.items()
            if g.status in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS) and gid != exclude_goal_id
        ]

        if not pending_goals:
            return None

        priority_order = {
            GoalPriority.HIGHEST: 0,
            GoalPriority.HIGH: 1,
            GoalPriority.MEDIUM: 2,
            GoalPriority.LOW: 3,
        }

        pending_goals.sort(
            key=lambda x: (
                priority_order.get(x[1].priority, 2),
                -(x[1].importance if x[1].importance is not None else 0.5),
            )
        )

        return pending_goals[0][0]
