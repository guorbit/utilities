from sphinx.addnodes import toctree as addnodes_toctree


def sub_package(ref):
    last_part = ref.split('/')[-1]
    return last_part

def process_toctree(app, doctree, docname):
    print(f"process_toctree called for doc: {docname}")
    for toctree_node in doctree.traverse(lambda node: isinstance(node, addnodes_toctree)):
        print(f"Found toctree in doc: {docname}")
        print("Before:", toctree_node['entries'])
        toctree_node['maxdepth'] = 1
        new_entries = []
        for title, ref in toctree_node['entries']:
            if title:
                new_title = "Auto: " + title
            else:
                # Check if ref contains multiple parts separated by dots
                if '.' in ref:
                    display_ref = ref.split('.')[-1]  # Take only the last part for display
                    new_title = sub_package(display_ref)
                else:
                    new_title = sub_package(ref)
            new_entries.append((new_title, ref))
        toctree_node['entries'] = new_entries
        print("After:", toctree_node['entries'])

def setup(app):
    app.connect('doctree-resolved', process_toctree)
