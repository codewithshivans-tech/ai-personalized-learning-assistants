import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class PersonalizedLearningAssistant:
    """AI-powered Personalized Learning Assistant for adaptive paths,
    recommendations, progress gamification, and learner clustering."""

    def __init__(self, state_file="learning_assistant_state.json"):
        self.state_file = state_file
        self.modules = self._build_modules()
        self.student_profiles = {}
        self.students_df = self._create_sample_students(n=20)
        self._initialize_profiles()
        self.load_state()

    def _build_modules(self):
        return [
            {
                "id": "M1",
                "title": "Python & Data Foundations",
                "topic": "python",
                "level": "Beginner",
                "duration_hrs": 4.0,
                "prereq": None,
                "points": 100,
            },
            {
                "id": "M2",
                "title": "NumPy & Pandas Data Wrangling",
                "topic": "data_science",
                "level": "Beginner",
                "duration_hrs": 5.0,
                "prereq": "M1",
                "points": 120,
            },
            {
                "id": "M3",
                "title": "Data Visualization & EDA",
                "topic": "data_science",
                "level": "Intermediate",
                "duration_hrs": 4.5,
                "prereq": "M2",
                "points": 130,
            },
            {
                "id": "M4",
                "title": "Supervised ML & Regression",
                "topic": "machine_learning",
                "level": "Intermediate",
                "duration_hrs": 6.0,
                "prereq": "M2",
                "points": 160,
            },
            {
                "id": "M5",
                "title": "Classification & Model Evaluation",
                "topic": "machine_learning",
                "level": "Intermediate",
                "duration_hrs": 6.0,
                "prereq": "M4",
                "points": 170,
            },
            {
                "id": "M6",
                "title": "Clustering & Unsupervised Learning",
                "topic": "machine_learning",
                "level": "Advanced",
                "duration_hrs": 5.5,
                "prereq": "M3",
                "points": 180,
            },
            {
                "id": "M7",
                "title": "Intro to Neural Networks",
                "topic": "deep_learning",
                "level": "Advanced",
                "duration_hrs": 7.0,
                "prereq": "M5",
                "points": 200,
            },
            {
                "id": "M8",
                "title": "End-to-End AI Capstone Project",
                "topic": "capstone",
                "level": "Advanced",
                "duration_hrs": 8.0,
                "prereq": "M6",
                "points": 250,
            },
        ]

    def _create_sample_students(self, n=20, seed=42):
        rng = np.random.default_rng(seed)
        topics = ["python", "data_science", "machine_learning", "deep_learning"]
        styles = ["visual", "hands_on", "reading"]

        rows = []
        for i in range(1, n + 1):
            sid = f"S{i:03d}"
            quiz_avg = round(float(rng.uniform(45, 96)), 1)
            study_hrs = round(float(rng.uniform(3.0, 22.0)), 1)
            completion_rate = round(float(rng.uniform(0.15, 0.95)), 2)
            streak_days = int(rng.integers(1, 25))
            interest = str(rng.choice(topics))
            style = str(rng.choice(styles))
            rows.append(
                {
                    "student_id": sid,
                    "name": f"Learner_{i:02d}",
                    "quiz_avg": quiz_avg,
                    "weekly_study_hrs": study_hrs,
                    "completion_rate": completion_rate,
                    "streak_days": streak_days,
                    "interest_topic": interest,
                    "learning_style": style,
                }
            )
        return pd.DataFrame(rows)

    def _initialize_profiles(self):
        for _, row in self.students_df.iterrows():
            sid = row["student_id"]
            completed = ["M1"] if row["completion_rate"] >= 0.25 else []
            if row["completion_rate"] >= 0.50:
                completed.append("M2")
            if row["completion_rate"] >= 0.75:
                completed.append("M4")

            points = len(completed) * 120 + int(row["streak_days"] * 10)
            badges = []
            if row["streak_days"] >= 7:
                badges.append("7-Day Streak")
            if row["quiz_avg"] >= 85:
                badges.append("Quiz Ace")
            if len(completed) >= 2:
                badges.append("Fast Starter")

            self.student_profiles[sid] = {
                "student_id": sid,
                "name": row["name"],
                "quiz_avg": float(row["quiz_avg"]),
                "weekly_study_hrs": float(row["weekly_study_hrs"]),
                "completion_rate": float(row["completion_rate"]),
                "streak_days": int(row["streak_days"]),
                "interest_topic": row["interest_topic"],
                "learning_style": row["learning_style"],
                "completed_modules": completed,
                "points": int(points),
                "badges": badges,
            }

    def generate_learning_path(self, student_id):
        profile = self.student_profiles.get(student_id)
        if not profile:
            raise ValueError(f"Unknown student_id: {student_id}")

        completed_set = set(profile["completed_modules"])
        path = []
        for mod in self.modules:
            if mod["id"] in completed_set:
                continue
            prereq = mod["prereq"]
            prereq_met = (prereq is None) or (prereq in completed_set)
            if prereq_met:
                path.append(mod)
        return path

    def recommend_content(self, student_id, top_n=3):
        profile = self.student_profiles.get(student_id)
        if not profile:
            raise ValueError(f"Unknown student_id: {student_id}")

        completed_set = set(profile["completed_modules"])
        interest = profile["interest_topic"]
        quiz_avg = profile["quiz_avg"]

        scored = []
        for mod in self.modules:
            if mod["id"] in completed_set:
                continue

            prereq = mod["prereq"]
            if prereq is not None and prereq not in completed_set:
                continue

            score = 50.0
            if mod["topic"] == interest:
                score += 25.0

            if quiz_avg < 65 and mod["level"] == "Beginner":
                score += 20.0
            elif 65 <= quiz_avg < 82 and mod["level"] in ("Beginner", "Intermediate"):
                score += 15.0
            elif quiz_avg >= 82 and mod["level"] in ("Intermediate", "Advanced"):
                score += 20.0

            score += mod["points"] * 0.05
            scored.append((round(score, 2), mod))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, **m} for s, m in scored[:top_n]]

    def record_progress(self, student_id, module_id, quiz_score):
        profile = self.student_profiles.get(student_id)
        if not profile:
            raise ValueError(f"Unknown student_id: {student_id}")

        mod = next((m for m in self.modules if m["id"] == module_id), None)
        if not mod:
            raise ValueError(f"Unknown module_id: {module_id}")

        if module_id not in profile["completed_modules"]:
            profile["completed_modules"].append(module_id)
            profile["points"] += mod["points"]

        if quiz_score >= 80:
            profile["streak_days"] += 1
            profile["points"] += 25

        old_avg = profile["quiz_avg"]
        profile["quiz_avg"] = round((old_avg + float(quiz_score)) / 2.0, 1)
        profile["completion_rate"] = round(
            len(profile["completed_modules"]) / len(self.modules), 2
        )

        if profile["streak_days"] >= 7 and "7-Day Streak" not in profile["badges"]:
            profile["badges"].append("7-Day Streak")
        if profile["quiz_avg"] >= 85 and "Quiz Ace" not in profile["badges"]:
            profile["badges"].append("Quiz Ace")
        if len(profile["completed_modules"]) >= 3 and "Module Explorer" not in profile["badges"]:
            profile["badges"].append("Module Explorer")

        self._sync_dataframe_row(student_id)
        return profile

    def _sync_dataframe_row(self, student_id):
        p = self.student_profiles[student_id]
        mask = self.students_df["student_id"] == student_id
        if mask.any():
            self.students_df.loc[mask, "quiz_avg"] = p["quiz_avg"]
            self.students_df.loc[mask, "completion_rate"] = p["completion_rate"]
            self.students_df.loc[mask, "streak_days"] = p["streak_days"]

    def cluster_learners(self, n_clusters=3):
        features = ["quiz_avg", "weekly_study_hrs", "completion_rate", "streak_days"]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.students_df[features])

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)

        clustered_df = self.students_df.copy()
        clustered_df["cluster"] = labels
        return clustered_df

    def generate_report(self, student_id):
        profile = self.student_profiles.get(student_id)
        if not profile:
            raise ValueError(f"Unknown student_id: {student_id}")

        next_path = self.generate_learning_path(student_id)
        recs = self.recommend_content(student_id, top_n=3)

        lines = [
            f"=== Learner Report: {profile['name']} ({student_id}) ===",
            f"Interest Topic : {profile['interest_topic']} | Style: {profile['learning_style']}",
            f"Quiz Average   : {profile['quiz_avg']}% | Completion Rate: {profile['completion_rate'] * 100:.0f}%",
            f"Study / Week   : {profile['weekly_study_hrs']} hrs | Streak: {profile['streak_days']} days",
            f"Points & Badges: {profile['points']} pts | Badges: {', '.join(profile['badges']) or 'None'}",
            f"Completed Mods : {', '.join(profile['completed_modules']) or 'None'}",
            "Available Next Modules:",
        ]
        for m in next_path:
            lines.append(f"  - [{m['id']}] {m['title']} ({m['level']}, {m['duration_hrs']}h, +{m['points']} pts)")

        lines.append("Top Recommended Content:")
        for r in recs:
            lines.append(f"  - [{r['id']}] {r['title']} (Score: {r['score']}, Topic: {r['topic']})")

        return "\n".join(lines)

    def save_state(self):
        payload = {
            "last_updated": datetime.now().isoformat(),
            "student_profiles": self.student_profiles,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return self.state_file

    def load_state(self):
        if not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_profiles = data.get("student_profiles", {})
            for sid, prof in saved_profiles.items():
                if sid in self.student_profiles:
                    self.student_profiles[sid].update(prof)
                    self._sync_dataframe_row(sid)
            return True
        except (json.JSONDecodeError, OSError):
            return False


def main():
    assistant = PersonalizedLearningAssistant(state_file="learning_assistant_state.json")

    print("--- Sample Learners (Top 6) ---")
    print(assistant.students_df.head(6).to_string(index=False))

    print("\n--- Learner Clustering (K-Means) ---")
    clustered = assistant.cluster_learners(n_clusters=3)
    summary = clustered.groupby("cluster")[
        ["quiz_avg", "weekly_study_hrs", "completion_rate", "streak_days"]
    ].mean().round(2)
    print(summary.to_string())

    for sid in ["S001", "S002", "S003", "S004"]:
        print("\n" + assistant.generate_report(sid))

    print("\n--- Recording Progress for S001 on M2 ---")
    updated = assistant.record_progress("S001", "M2", quiz_score=90)
    print(
        f"S001 Updated -> Points: {updated['points']}, "
        f"Streak: {updated['streak_days']}, Quiz Avg: {updated['quiz_avg']}"
    )

    saved_path = assistant.save_state()
    print(f"\nState saved successfully to: {saved_path}")


if __name__ == "__main__":
    main()
