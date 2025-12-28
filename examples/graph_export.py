from blockether_foundation.graph import GraphDatabase

graph = GraphDatabase.load_from_file("knowledge_graph.json")

graph.export_to_html("knowledge_graph.html", title="Knowledge Graph")
