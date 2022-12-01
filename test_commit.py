import argparse
import os


def test_commit(message=None): #pragma: no cover

    run_tests_command = "coverage run -m testing.test_all"
    make_report_command = "coverage report > coverage.txt"

    print(f"running on command line: \n  {run_tests_command}")
    os.system(run_tests_command)
    print(f"running on command line: \n  {make_report_command}")
    os.system(make_report_command)

    with open("coverage.txt", "r") as f:
        for line in f.readlines():
            if "TOTAL" in line:
                summary = line

    git_add_command = "git add coverage.txt"
    commit_command = f"git commit -m 'test commit summary: {summary}' "

    if message is not None:
        commit_command += f"-m '{message}'"

    os.system(git_add_command)
    os.system(commit_command)

if __name__ == "__main__": #pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str, default=None,\
            help="optional, additional message to add to commit")

    args = parser.parse_args()

    test_commit(message=args.message)

