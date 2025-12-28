"""Shared color palette for blockether-foundation visualizations."""

from __future__ import annotations

# Primary brand colors
PRIMARY_YELLOW = "#FFCC00"
PRIMARY_YELLOW_DARK = "#E6B800"
PRIMARY_YELLOW_LIGHT = "#FFE066"

# Neutral colors
WHITE = "#FFFFFF"
GRAY_50 = "#F9FAFB"
GRAY_100 = "#F3F4F6"
GRAY_200 = "#E5E7EB"
GRAY_300 = "#D1D5DB"
GRAY_400 = "#9CA3AF"
GRAY_500 = "#6B7280"
GRAY_600 = "#4B5563"
GRAY_700 = "#374151"
GRAY_800 = "#1F2937"
GRAY_900 = "#111827"

# Status colors
SUCCESS_GREEN = "#10B981"
SUCCESS_GREEN_LIGHT = "#D1FAE5"
WARNING_AMBER = "#F59E0B"
WARNING_AMBER_LIGHT = "#FEF3C7"
ERROR_RED = "#EF4444"
ERROR_RED_LIGHT = "#FEE2E2"
INFO_BLUE = "#3B82F6"
INFO_BLUE_LIGHT = "#DBEAFE"

# Chart/visualization colors (for distinct categories)
CHART_COLORS: list[str] = [
    "#3B82F6",  # Blue
    "#8B5CF6",  # Purple
    "#10B981",  # Green
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#84CC16",  # Lime
    "#F97316",  # Orange
    "#6366F1",  # Indigo
]

# Entity type colors for graph visualization
ENTITY_TYPE_COLORS: dict[str, str] = {
    "person": "#3B82F6",
    "organization": "#8B5CF6",
    "location": "#10B981",
    "event": "#F59E0B",
    "document": "#6366F1",
    "concept": "#EC4899",
    "object": "#14B8A6",
    "creature": "#F97316",
    "date": "#84CC16",
    "attachment": "#06B6D4",
    "example": "#A855F7",
    "rule": "#EF4444",
    "pattern": "#22C55E",
    "mode": "#0EA5E9",
    "schema": "#D946EF",
    "abbreviation": "#F472B6",
    "reference": "#64748B",
    "memory": "#FB923C",
    "situation": "#4ADE80",
    "fact": "#FBBF24",
}
