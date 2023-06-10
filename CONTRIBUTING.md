# Contributing to segment-anything
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. **Fork the Repository**: Start by forking the Segment-Anything repository to your own GitHub account.
    - A fork is a copy of a repository. Forking a repository allows you to freely experiment with changes without affecting the original project. In the upper right of the main repo you may select fork and create a copy for your profile.

2. **Create a New Branch**: From the `main` branch, create a new branch to contain your changes.
    - Creating a new branch is done with the following command in your forked copy:

        ```
        git checkout -b <branch-name>
        ```

3. **Make Your Changes**: Implement your changes in this branch. 

4. **Write Tests**: If you've added code that should be tested, ensure you include appropriate tests.

5. **Update the Documentation**: If your changes include updates to APIs, make sure you update the corresponding documentation.

6. **Run Tests**: Ensure that all existing and new tests pass. 

7. **Lint Your Code**: Linting is the process of checking your source code for programmatic and stylistic errors. Make sure your code adheres to our styling conventions. You can do this by using the `linter.sh` script in the project's root directory. 

    - Run the following command: 

        ```
        pip install black==23.* isort==5.12.0 flake8 mypy
        ```

    - Run Linter Script: After you've installed the required tools, you can run the linter.sh script located in the root directory of the project. Navigate to the project's root directory in your terminal or command line and run the following command:

        ```
        ./linter.sh
        ```

    - This command will start the linting process. The script will check your code and report any stylistic or programmatic errors it finds.
    - **Note** - Please note that you might need to make the linter script executable before you can run it. You can do this with the following command:

        ```
        chmod +x linter.sh
        ```

8. **Sign the CLA**: If you haven't already, complete the Contributor License Agreement ("CLA"). More information below.

9. **Submit a Pull Request**: Upon completion of the above steps, you can create a pull request (PR). Go to the GitHub page for Segment Anything's original repository, navigate to the "Pull requests" tab, and then click the "New pull request" button. On the next page, click the "compare across forks" link, then select your fork and the branch with your changes. Fill out the form describing your changes and then click "Create pull request."

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to segment-anything, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
