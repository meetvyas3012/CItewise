"""
Example usage of the Multi-Document Research Assistant
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Example usage of the RAG system"""

    print("🚀 Multi-Document Research Assistant Example")
    print("=" * 50)

    try:
        # Import the system components
        from multi_doc_rag import (
            quick_start, add_document, ask_question,
            document_processor, db_manager, embedding_manager
        )

        # Initialize the system
        print("\n1. Initializing system...")
        if not quick_start():
            print("❌ Failed to initialize system")
            return

        # Check if we have any documents
        documents = db_manager.list_documents()
        print(f"\n2. Current documents in system: {len(documents)}")

        if len(documents) == 0:
            print("\n📄 No documents found. To add documents:")
            print("   1. Place PDF, TXT, DOCX, HTML, or MD files in the 'data/documents' folder")
            print("   2. Use the web interface: streamlit run multi_doc_rag/ui/streamlit_app.py")
            print("   3. Use the CLI: python -m multi_doc_rag.cli add /path/to/document")
            print("   4. Use the Python API: add_document('/path/to/document')")

            # Create a sample document for demonstration
            sample_doc_path = project_root / "data" / "documents" / "sample.txt"
            sample_doc_path.parent.mkdir(parents=True, exist_ok=True)

            sample_content = """
            Artificial Intelligence and Machine Learning

            Artificial Intelligence (AI) is a broad field of computer science that aims to create 
            systems capable of performing tasks that typically require human intelligence. These 
            tasks include learning, reasoning, problem-solving, perception, and language understanding.

            Machine Learning (ML) is a subset of AI that focuses on the development of algorithms 
            and statistical models that enable computers to improve their performance on a specific 
            task through experience, without being explicitly programmed for every scenario.

            Deep Learning is a subset of machine learning that uses neural networks with multiple 
            layers (deep neural networks) to model and understand complex patterns in data. It has 
            been particularly successful in areas such as image recognition, natural language 
            processing, and speech recognition.

            Applications of AI include:
            - Autonomous vehicles
            - Medical diagnosis
            - Recommendation systems
            - Natural language processing
            - Computer vision
            - Robotics

            The field continues to evolve rapidly, with new breakthroughs in areas such as 
            generative AI, reinforcement learning, and quantum machine learning.
            """

            with open(sample_doc_path, 'w') as f:
                f.write(sample_content)

            print(f"\n📝 Created sample document: {sample_doc_path}")
            print("\n3. Adding sample document to system...")

            try:
                doc_id = add_document(sample_doc_path)
                print(f"✅ Document added successfully! ID: {doc_id}")

                # Update documents list
                documents = db_manager.list_documents()

            except Exception as e:
                print(f"❌ Error adding document: {e}")
                return

        print(f"\n📊 System stats:")
        print(f"   Documents: {len(documents)}")
        for doc in documents:
            print(f"   - {doc['filename']} ({doc['chunk_count']} chunks)")

        # Example queries
        example_queries = [
            "What is artificial intelligence?",
            "What are the applications of AI?",
            "How does machine learning work?",
            "What is the difference between AI and machine learning?"
        ]

        print("\n4. Example queries:")
        for i, query in enumerate(example_queries, 1):
            print(f"   {i}. {query}")

        # Ask a sample question
        print("\n5. Asking a sample question...")
        sample_query = "What is artificial intelligence and what are its applications?"

        try:
            print(f"\n🔍 Query: {sample_query}")
            print("   Searching and generating response...")

            response = ask_question(sample_query, k=3)

            print("\n💡 Response:")
            print("-" * 40)
            print(response.text)

            if response.sources:
                print(f"\n📚 Sources used: {len(response.sources)}")
                for source in response.sources:
                    print(f"   [{source['citation_number']}] {source['text_preview'][:100]}...")

            print(f"\n⚡ Performance:")
            print(f"   Generation time: {response.generation_time:.2f}s")
            print(f"   Model used: {response.model_used}")
            if response.confidence_score:
                print(f"   Confidence: {response.confidence_score:.1%}")

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return

        print("\n✅ Example completed successfully!")
        print("\n🌐 Next steps:")
        print("   1. Start web interface: streamlit run multi_doc_rag/ui/streamlit_app.py")
        print("   2. Add more documents through the web interface")
        print("   3. Try more complex queries")
        print("   4. Explore the analytics dashboard")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
