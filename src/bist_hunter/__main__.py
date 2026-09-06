from .morning_job import run_morning_job


if __name__ == "__main__":
    result = run_morning_job()
    print(result.status)
    print(result.message)
