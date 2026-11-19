# AI Personalized Learning Assistant (`AI-LT-002`)

Adaptive learning and personalized content recommendation system built in Python for the **AI & Data Science Internship** at **Data Alcott Systems**.
## Features Implemented
- **Student Profiling & Learning Style Detection:** Evaluates learner performance, engagement, and pace to align content with `Visual`, `Auditory`, `Kinesthetic`, and `Reading/Writing` styles.
- **Prerequisite-Aware Adaptive Learning Path:** Filters modules by learner skill level (`Beginner`, `Intermediate`, `Advanced`), enforces module prerequisites, and calculates total estimated study duration.
- **Content Recommendation Engine:** Ranks learning modules based on learning style alignment, difficulty match, and optional topic filtering.
- **Progress Tracking & Real-Time Adaptation:** Updates completion percentage, average scores, engagement, and skill/pace adaptation dynamically.
- **Gamification:** Awards XP points and milestone badges (`First Step Badge`, `Consistent Learner Badge`, `Mastery Star Badge`, `Streak & XP Champion`).
- **ML Learner Clustering (`Scikit-learn`):** Uses `StandardScaler` and `KMeans` to group learners into cohorts based on progress, scores, and engagement.
- **JSON State Persistence:** Saves and reloads student profiles and metrics using `learning_assistant_state.json` without requiring an external database.

## Tech Stack
- **Language:** Python 3.
- **Libraries:** `pandas`, `numpy`, `scikit-learn`
- **Storage:** JSON / in-memory Python data structures

## Setup & Execution
1. Clone the repository and navigate into the project folder:
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-folder>