from huggingface_hub import HfApi
import sys

token = sys.argv[1] if len(sys.argv) > 1 else ""
api = HfApi(token=token)

for repo in ["wzmmmm/plandiff-cross-att", "wzmmmm/plandiff-double-cross-6k"]:
    try:
        files = [f.rfilename for f in api.list_repo_files(repo)]
        print(f"{repo}: {files}")
    except Exception as e:
        print(f"{repo}: ERROR - {e}")
