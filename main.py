from dotenv import load_dotenv

load_dotenv()
from pprint import pprint

from graph.graph import app

question = "What is the document in the database about?"
inputs = {"question": question}

for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
print(value["generation"])   #["generation"]
