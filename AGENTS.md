
# Python coding standards
Use the black formatter to format any code before linting, testing or committing. Note that black doesn't format comment lines, you have to make sure that your comment lines stay within the 100 character limit/line.

## Linters
- Use pylint to lint code before submitting, and fix any errors & warnings in new or modified code. Don't overuse `# type: ignore` to "solve" linting problems. Use it only if the linter error is obviously superflous. Don't fix linting issues in existing code that you haven't changed, unless specifically instructed to do so.
- Run `bandit -c .bandit.yml -r .` to scan for security issues and fix any errors and warnings.

Linters are generally set up to ignore files in "tests/", so no need to lint tests, but still use the black formatter on them.
Only after linting issues are dealt with in new/modified code, run pytest, if tests exist for your changes.

# Bug fixing
Always create a test which replicates the bug you intend to fix as the first step, if the bug is not caught by any of the existing tests. 
Run the relevant test to confirm that it fails, before starting to fix the bug.

# New features, other non-bug code changes
Make sure to update any tests where the tested unit's expected behaviour is different because of the new changes.
Write unit tests for any new functions / modules which are created
