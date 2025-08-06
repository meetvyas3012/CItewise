import os
import sys

# Create the project structure
project_structure = {
    'multi_doc_rag': {
        '__init__.py': '',
        'core': {
            '__init__.py': '',
            'document_processor.py': '',
            'embeddings.py': '',
            'retrieval.py': '',
            'generation.py': '',
            'database.py': ''
        },
        'utils': {
            '__init__.py': '',
            'text_processing.py': '',
            'chunking.py': '',
            'evaluation.py': ''
        },
        'config': {
            '__init__.py': '',
            'settings.py': ''
        },
        'ui': {
            '__init__.py': '',
            'streamlit_app.py': '',
            'fastapi_app.py': ''
        }
    },
    'tests': {
        '__init__.py': '',
        'test_document_processor.py': '',
        'test_retrieval.py': '',
        'test_generation.py': ''
    },
    'data': {
        'documents': {},
        'embeddings': {},
        'models': {}
    },
    'requirements.txt': '',
    'docker-compose.yml': '',
    'Dockerfile': '',
    'README.md': '',
    'setup.py': ''
}

def create_directory_structure(structure, base_path=''):
    """Create directory structure recursively"""
    for name, content in structure.items():
        current_path = os.path.join(base_path, name) if base_path else name
        
        if isinstance(content, dict):
            # It's a directory
            os.makedirs(current_path, exist_ok=True)
            print(f"Created directory: {current_path}")
            create_directory_structure(content, current_path)
        else:
            # It's a file
            if not os.path.exists(current_path):
                with open(current_path, 'w') as f:
                    f.write(content)
                print(f"Created file: {current_path}")

# Create the project structure
print("Creating Multi-Document RAG Project Structure...")
create_directory_structure(project_structure)
print("\nProject structure created successfully!")

# List all created files and directories
def list_structure(path='.', indent=0):
    """List the created structure"""
    items = []
    try:
        for item in sorted(os.listdir(path)):
            if item.startswith('.'):
                continue
            item_path = os.path.join(path, item)
            indent_str = "  " * indent
            if os.path.isdir(item_path):
                items.append(f"{indent_str}{item}/")
                items.extend(list_structure(item_path, indent + 1))
            else:
                items.append(f"{indent_str}{item}")
    except PermissionError:
        pass
    return items

print("\nProject Structure:")
structure_list = list_structure()
for item in structure_list[:50]:  # Limit output
    print(item)