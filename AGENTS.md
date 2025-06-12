
# Python coding standards
Use the black formatter to format any code before linting, testing or committing. Note that black doesn't format comment lines, you have to make sure that your comment lines stay within the 100 character limit/line.
Use pylint (preferably) or pyright to lint code before submitting, and fix any errors & warnings in new or modified code. Don't overuse `# type: ignore` to "solve" linting problems. Use it only if the linter error is obviously superflous. Don't fix linting issues in existing code that you haven't changed, unless specifically instructed to do so.
Only after linting issues are dealt with in new/modified code, run pytest, if tests exist for your changes.
