"""Shared color palette and HTML templates for blockether-foundation visualizations."""

from __future__ import annotations

# ============================================================================
# COLOR PALETTE
# ============================================================================

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

# ============================================================================
# HTML TEMPLATES FOR REPORTS
# ============================================================================

# HTML head with CSS and JS libraries for reports
HTML_REPORT_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {{
            --primary-yellow: {PRIMARY_YELLOW};
            --primary-yellow-dark: {PRIMARY_YELLOW_DARK};
            --primary-yellow-light: {PRIMARY_YELLOW_LIGHT};
        }}

        * {{
            font-family: 'Atkinson Hyperlegible', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        body {{
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        .header-accent {{
            background: linear-gradient(90deg, {PRIMARY_YELLOW} 0%, {PRIMARY_YELLOW_LIGHT} 100%);
        }}

        .card {{
            background: {WHITE};
            border: 1px solid {GRAY_100};
            border-radius: 0;
        }}

        .card-accent {{
            border-left: 4px solid {PRIMARY_YELLOW};
        }}

        .score-ring {{
            stroke-linecap: round;
            transition: stroke-dasharray 0.5s ease;
        }}

        /* Accordion styles with full hover highlight */
        .accordion-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}

        .accordion-content.open {{
            max-height: 5000px;
        }}

        .accordion-icon {{
            transition: transform 0.3s ease;
        }}

        .accordion-icon.rotate-180 {{
            transform: rotate(180deg);
        }}

        .accordion-header {{
            transition: background-color 0.2s ease;
        }}

        .accordion-header:hover {{
            background-color: {GRAY_50} !important;
        }}

        .tooltip {{
            position: absolute;
            background: {GRAY_800};
            color: {WHITE};
            padding: 0.5rem 0.75rem;
            border-radius: 0;
            font-size: 0.75rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 100;
            max-width: 300px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}

        /* Flow arrow styles */
        .flow-arrow {{
            position: relative;
            padding-bottom: 3rem;
        }}

        .flow-arrow::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 2px;
            height: 2rem;
            background-color: {GRAY_300};
        }}

        /* Sticky step navigation - full width */
        .step-nav {{
            position: sticky;
            top: 85px;
            z-index: 40;
            background: {WHITE};
            border-bottom: 1px solid {GRAY_200};
            margin-bottom: 1.5rem;
            padding: 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}

        .step-nav-inner {{
            display: flex;
            align-items: center;
            justify-content: space-around;
            gap: 0;
            padding: 0.75rem 24px;
            overflow-x: auto;
            scrollbar-width: none;
            max-width: 100%;
        }}

        .step-nav-inner::-webkit-scrollbar {{
            display: none;
        }}

        .step-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.25rem;
            padding: 0.5rem 0.75rem;
            font-size: 0.7rem;
            font-weight: 500;
            color: {GRAY_500};
            white-space: nowrap;
            transition: all 0.2s ease;
            cursor: pointer;
            text-align: center;
            flex: 1;
        }}

        .step-item:hover {{
            color: {GRAY_700};
        }}

        .step-item.active {{
            color: {GRAY_900};
            font-weight: 700;
        }}

        .step-number {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 0;
            font-size: 0.625rem;
            font-weight: 600;
            background: {GRAY_200};
            color: {GRAY_600};
            margin-bottom: 0.125rem;
        }}

        .step-item.active .step-number {{
            background: {GRAY_900};
            color: {WHITE};
        }}

        /* Consensus table - full width */
        .consensus-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}

        .consensus-table th {{
            background: {GRAY_100};
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            color: {GRAY_700};
            border-bottom: 2px solid {GRAY_200};
        }}

        .consensus-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid {GRAY_200};
            color: {GRAY_700};
        }}

        .consensus-table tr:last-child td {{
            border-bottom: none;
        }}

        .consensus-table tr:hover td {{
            background: {GRAY_50};
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.125rem 0.5rem;
            border-radius: 0;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        .status-badge.agreement {{
            background: {SUCCESS_GREEN_LIGHT};
            color: {SUCCESS_GREEN};
        }}

        .status-badge.conflict {{
            background: {WARNING_AMBER_LIGHT};
            color: {WARNING_AMBER};
        }}

        .status-badge.uncertainty,
        .status-badge.insight {{
            background: {GRAY_100};
            color: {GRAY_700};
        }}

        /* Criteria badge */
        .criteria-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 0;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        /* Insight weight badge */
        .insight-weight {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 0;
            font-size: 1rem;
            font-weight: 700;
            color: {GRAY_300};
        }}

        /* Raw output pre */
        .raw-output {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Atkinson Hyperlegible', monospace;
            font-size: 0.875rem;
            line-height: 1.625;
            color: {GRAY_800};
        }}

        /* Section transition */
        .report-section {{
            animation: fadeIn 0.3s ease-in-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: {GRAY_100};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {GRAY_400};
            border-radius: 0;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {GRAY_500};
        }}
    </style>
</head>"""

# HTML body opening with header - template with placeholders
HTML_REPORT_HEADER_TEMPLATE = """<body class="bg-gray-50 min-h-screen">
    <div id="tooltip" class="tooltip"></div>

    <header class="bg-white border-b border-gray-100 sticky top-0 z-50 shadow-sm">
        <div class="header-accent h-1"></div>
        <div class="max-w-6xl mx-auto px-4 py-4">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-xl font-bold text-gray-900">{title}</h1>
                    <p class="text-sm text-gray-500">{subtitle}</p>
                </div>
                <div class="flex items-center gap-6">
                    <div class="text-center">
                        <div class="text-2xl font-bold" style="color: {confidence_color}">{confidence_score}</div>
                        <div class="text-xs text-gray-500 uppercase tracking-wider">Confidence</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold" style="color: {judge_color}">{judge_score}</div>
                        <div class="text-xs text-gray-500 uppercase tracking-wider">Judge Score</div>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main>"""

# HTML footer
HTML_REPORT_FOOTER = """
    </main>

    <footer class="border-t border-gray-200 bg-white py-6 mt-12">
        <div class="max-w-6xl mx-auto px-4 text-center">
            <p class="text-xs text-gray-500">{timestamp} | <a href="https://blockether.com" class="text-gray-600 hover:text-gray-900">blockether.com</a></p>
        </div>
    </footer>

    <script>
        // Initialize all accordions
        document.querySelectorAll('.accordion-header').forEach(header => {{
            header.addEventListener('click', function() {{
                const content = this.nextElementSibling;
                const icon = this.querySelector('.accordion-icon');
                content.classList.toggle('open');
                icon.classList.toggle('rotate-180');
            }});
        }});

        // Step navigation highlighting on scroll
        const steps = document.querySelectorAll('.step-item');
        const sections = document.querySelectorAll('.report-section');

        const observerOptions = {{
            root: null,
            rootMargin: '-20% 0px -70% 0px',
            threshold: 0
        }};

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const id = entry.target.getAttribute('data-step');
                    steps.forEach(step => {{
                        step.classList.remove('active');
                        if (step.getAttribute('data-step-target') === id) {{
                            step.classList.add('active');
                        }}
                    }});
                }}
            }});
        }}, observerOptions);

        sections.forEach(section => observer.observe(section));

        // Step navigation click handling
        steps.forEach(step => {{
            step.addEventListener('click', function() {{
                const targetId = this.getAttribute('data-step-target');
                const targetSection = document.querySelector(`[data-step="${{targetId}}"]`);
                if (targetSection) {{
                    targetSection.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }});
        }});
    </script>
</body>
</html>"""

# D3.js chart initialization script - monochrome version
D3_CHART_INIT_SCRIPT = """    <script>
        const models = {models_json};
        const chartColors = {chart_colors_json};

        // Generate grayscale colors for models
        const numModels = models.length;
        const grayColors = models.map((_, i) => {{
            const lightness = 80 - (i * 50 / Math.max(numModels - 1, 1));
            return `hsl(0, 0%, ${{lightness}}%)`;
        }});

        const width = 140, height = 140, radius = Math.min(width, height) / 2;

        const svg = d3.select("#model-chart")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", `translate(${{width/2}}, ${{height/2}})`);

        const pie = d3.pie().value(d => d.importance).sort(null);
        const arc = d3.arc().innerRadius(radius * 0.5).outerRadius(radius * 0.9);

        const tooltip = d3.select("#tooltip");

        const arcs = svg.selectAll("path")
            .data(pie(models))
            .join("path")
            .attr("d", arc)
            .attr("fill", (d, i) => grayColors[i])
            .attr("stroke", "{WHITE}")
            .attr("stroke-width", 2)
            .style("cursor", "pointer")
            .on("mouseover", function(event, d) {{
                d3.select(this).attr("opacity", 0.8);
                tooltip.style("opacity", 1)
                    .html(`<strong>${{d.data.name}}</strong><br/>Importance: ${{(d.data.importance * 100).toFixed(0)}}%<br/>Incorporated: ${{d.data.incorporated}}<br/>Rejected: ${{d.data.rejected}}`);
            }})
            .on("mousemove", function(event) {{
                tooltip.style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", function() {{
                d3.select(this).attr("opacity", 1);
                tooltip.style("opacity", 0);
            }});

        const legend = d3.select("#model-legend");
        models.forEach((m, i) => {{
            legend.append("div")
                .attr("class", "flex items-center gap-2")
                .html(`
                    <div class="w-3 h-3 rounded-full flex-shrink-0" style="background-color: ${{grayColors[i]}}"></div>
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-medium text-gray-900 truncate">${{m.name}}</div>
                        <div class="text-xs text-gray-500">${{m.perspective}}</div>
                    </div>
                    <div class="text-sm font-bold text-gray-700">${{(m.importance * 100).toFixed(0)}}%</div>
                `);
        }});
    </script>"""
