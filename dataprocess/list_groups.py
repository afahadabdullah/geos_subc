from arraylake import Client

client = Client()
repo = client.get_repo("umd/subc")
print("Available Groups:")
for group in repo.groups:
    print(f"- {group.name}")
