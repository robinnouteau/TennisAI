
def get_git_hash():
    sha = ''
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except:  # noqa: E722
        pass
    return sha
