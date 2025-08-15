import os

def list_files(startpath):
    """Lists files and directories recursively in a tree structure."""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

# Specify the root directory of your repository
repository_root = '.'  # Use '.' for the current directory

list_files(repository_root)