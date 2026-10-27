"""Manage login accounts.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m backend.auth_admin generate-secret
    .\\.venv\\Scripts\\python.exe -m backend.auth_admin add-user alice --role admin
    .\\.venv\\Scripts\\python.exe -m backend.auth_admin add-user recruiter --role viewer
    .\\.venv\\Scripts\\python.exe -m backend.auth_admin list-users
    .\\.venv\\Scripts\\python.exe -m backend.auth_admin remove-user recruiter

Passwords are typed at a hidden prompt (never as arguments, which would end up in shell
history) and stored only as salted scrypt hashes in AUTH_USERS_FILE (default
data/users.json, which is gitignored). Adding an existing user resets their password and
role. Changes take effect immediately, including for already logged-in users: a removed
user's tokens stop working.
"""

import argparse
import getpass
import sys

from backend.auth import MIN_PASSWORD_LENGTH, ROLES, UserStore, generate_secret_key, validate_new_credentials
from backend.config import get_settings


def main(argv: list[str] | None = None, prompt=getpass.getpass) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate-secret", help="print a new random AUTH_SECRET_KEY")
    add = commands.add_parser("add-user", help="create a user or reset their password/role")
    add.add_argument("username")
    add.add_argument("--role", choices=ROLES, required=True)
    commands.add_parser("list-users")
    remove = commands.add_parser("remove-user")
    remove.add_argument("username")
    args = parser.parse_args(argv)

    if args.command == "generate-secret":
        print("Add this line to .env (keep it secret; changing it logs everyone out):")
        print(f"AUTH_SECRET_KEY={generate_secret_key()}")
        return 0

    store = UserStore(get_settings().auth_users_path)
    if args.command == "list-users":
        users = store.list_users()
        for name, role in sorted(users.items()):
            print(f"{name:32} {role}")
        if not users:
            print(f"No users in {store.path}.")
        return 0

    if args.command == "remove-user":
        if store.remove_user(args.username.lower()):
            print(f"Removed {args.username}. Their existing sessions no longer work.")
            return 0
        print(f"No user '{args.username}'.", file=sys.stderr)
        return 1

    username = args.username.strip().lower()
    password = prompt(f"Password for {username} (min {MIN_PASSWORD_LENGTH} characters): ")
    if password != prompt("Repeat the password: "):
        print("The passwords do not match; nothing was changed.", file=sys.stderr)
        return 1
    try:
        validate_new_credentials(username, password, args.role)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    existed = username in store.list_users()
    store.add_user(username, password, args.role)
    print(f"{'Updated' if existed else 'Created'} {username} ({args.role}) in {store.path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
