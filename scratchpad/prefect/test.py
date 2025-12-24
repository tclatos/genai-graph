import os

from prefect import flow, task

# Disable proxy for local connections to avoid timeout issues
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"


@task
def get_customer_ids() -> list[str]:
    # Fetch customer IDs from a database or API
    return [f"customer{n}" for n in range(1, 11)]


@task
def process_customer(customer_id: str) -> str:
    # Process a single customer
    return f"Processed {customer_id}"


@flow(log_prints=True)
def main() -> list[str]:
    customer_ids = get_customer_ids()
    # Process each customer in parallel using list comprehension with submit
    results = [process_customer.submit(cid) for cid in customer_ids]
    # Wait for all results
    return [r.result() for r in results]


if __name__ == "__main__":
    result = main()
    print(f"Results: {result}")
# if __name__ == "__main__":
#     main.serve(name="my-first-deployment")
