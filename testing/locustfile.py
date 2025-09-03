from locust import HttpUser, task, between
import os


API_TOKEN = os.getenv("LOCUST_API_BEARER")


class IFRS9ApiUser(HttpUser):
    wait_time = between(0.1, 1.5)

    def on_start(self):
        self.headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

    @task(3)
    def health(self):
        self.client.get("/health")

    @task(2)
    def status(self):
        self.client.get("/api/v1/status", headers=self.headers)

    @task(5)
    def predict(self):
        payload = {
            "loan_amount": 120000,
            "current_balance": 80000,
            "interest_rate": 4.2,
            "credit_score": 710,
            "dti_ratio": 0.32,
            "employment_length": 24,
            "days_past_due": 0,
        }
        self.client.post("/api/v1/predict/risk", json=payload, headers=self.headers)

