# GitHub identity and Lambda synchronization

Repository: https://github.com/xiangli-damien/openact

## Checkouts and authentication

| Purpose | Directory | Remote |
| --- | --- | --- |
| Local development | `/Users/lixiang/Projects/openact` | `git@github.com:xiangli-damien/openact.git` |
| Active GPU code, logs and results | `/lambda/nfs/dami/openact` (`~/dami/openact`) | Same repository |
| Older GPU checkout containing the virtual environment | `/home/ubuntu/openact` | Same repository |

The active virtual environment resolves OpenAct packages from the dami checkout.
The home checkout's `.venv` remains on instance storage. Open `~/dami/openact` in
Cursor for current work.

Both local and forwarded Lambda SSH authentication identify as `xiangli-damien`.
Git authentication and commit authorship are separate: authentication controls
access, while the email embedded in each commit determines account attribution.

On 2026-09-16, repository-local settings were applied to all three checkouts:

```ini
[user]
    name = Xiang Li
    email = 121291188+xiangli-damien@users.noreply.github.com
    useConfigOnly = true
[pull]
    ff = only
```

The GitHub private address uses the authenticated account ID and login. These
settings override the local machine's global Gitee identity for this repository.
Other repositories keep their existing settings.

## Code update procedure

After reviewing and testing a local change:

```sh
# Local development checkout
git status --short
git add <reviewed-files>
git commit -m '<description>'
git push origin main

# GPU active checkout
ssh gpu2 'cd /lambda/nfs/dami/openact && git pull --ff-only origin main'
```

`git fetch origin` followed by `git merge --ff-only origin/main` is the equivalent
two-step procedure used in the recent deployment reflog. It retrieves committed
code from GitHub. File copies used for local-to-dami activation transfers are
separate from code deployment; generated activations stay outside Git.

Before updating code used by active jobs, compare changed paths with their pinned
`job_plan.json` code hashes. Documentation-only changes can be deployed without
restarting collection. Changes to pinned implementation files require a separate
checkout or a deliberate restart plan. Do not reinstall the active environment
or restart workers as part of a documentation/identity update.

The home checkout can also be refreshed with
`git -C /home/ubuntu/openact pull --ff-only origin main`, but use the dami checkout
as the working directory. Fast-forward-only pulls reject unexpected divergence.

## Historical attribution repair (2026-09-16)

Before correction, the 23 commits from `7c3593f` through `7a8171b` inherited
`13487894+xiangli2001@user.noreply.gitee.com`, with no associated GitHub author or
committer account. Commit `93c9f6c` used the placeholder `your-email@example.com`.
Commit `9925f8a` already mapped correctly to the owner.

After the user's approval, the 24 incorrect author/committer identities were
corrected. All 26 pre-existing commits were checked individually: their file
trees, messages, and author/committer timestamps remained identical. The repaired
tip is `bd182eb8f68f9b9a9be1fe8cfea2422e17f42960`; GitHub's API associates all 26
authors and committers with `xiangli-damien`.

An atomic push preserved the old history in branch
`codex/backup-before-authorship-20260916` while updating `main` with an explicit
lease on its previous head (`c99f6009644f070594698c231e0305a92dbe4ac6`). Local and
both Lambda checkout references were moved only after verifying identical trees
and clean workspaces. Working files and running collection processes were not
restarted or rewritten for this metadata repair.

The backup branch remains on GitHub and all three checkouts. A local bundle is
also retained at `.git/authorship-review-20260916/original-main.bundle`.
[git_authorship_migration.json](git_authorship_migration.json) records every
old-to-new commit ID. Existing collection manifests retain their original IDs;
those commits remain reachable through the backup, preserving reproducibility.
Normal future updates continue through push and fast-forward-only pull.

GitHub reference: [commit email addresses](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).
