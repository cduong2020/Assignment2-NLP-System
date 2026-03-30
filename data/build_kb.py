import json
import os


def build_single_knowledge_base():
    with open("data/knowledge_base.txt", "w", encoding="utf-8") as kb:
        # 1. Add Faculty
        with open("data/faculty_info.json", "r") as f:
            faculty = json.load(f)
            for p in faculty:
                kb.write(
                    f"FACULTY INFO: {p['name']} is a {p['role']}. Interests: {', '.join(p['interests'])}. Profile: {p['url']}\n")

        # 2. Add Papers
        paper_path = "data/raw_papers/"
        for filename in os.listdir(paper_path):
            if filename.endswith(".txt"):
                with open(os.path.join(paper_path, filename), "r", encoding="utf-8") as pf:
                    kb.write(f"\nRESEARCH PAPER: {pf.read()}\n")


if __name__ == "__main__":
    build_single_knowledge_base()