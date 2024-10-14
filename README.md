# CAIS_DigitalWellbeing
In this project, the objective is to develop a machine learning learning model to improve the health and well-being of people through an analysis of their screentime. The dataset comprises of detailed fields to monitor smartphone usage, and the goal is to improve digital well-being (especially for college students). 

Dataset Information:
- Demographics:
    - uid:	Unique user ID consisting of alphanumeric characters **(string)**
    - gender: Self-reported gender of the participant **(string)**
    - race:	Self-reported race of the participant **(string)**
      
- COVID-19 Survey Responses:
    - uid:	Unique user ID consisting of alphanumeric characters **(string)**
    - date: Date in the format of YYYYMMDD **(int)**
    - questions: 10 Questions about COVID-19 answered on a scale of 1-7 **(int)**
      
- General Questions Survey Responses:
    - uid:	Unique user ID consisting of alphanumeric characters **(string)**
    - date: Date in the format of YYYYMMDD **(int)**
    - Photographic Affect Meter Score: The user chooses an image from 16 pictures and the valence and arousal are calculated **(int)**
          Rubric for PAM score: https://dl.acm.org/doi/10.1145/1978942.1979047 
    - behavior questions (phq4): 4 questions about behavior over the past two weeks on a scale of 0-3 **(int)**
    - behavior questions response time: amount of time spent on behavior questions (median & mean measured in seconds) **(int)**
    - phq4_score: sum of behavior questions scores **(int)**
    - social_level: question about user social level measured on a scale of 1-5 **(int)**
    - self esteem questions (sse3): 4 questions about user self esteem when responding to the survey measured on a scale of 1-5 **(int)**
    - self esteem questions response time: amount of time spent on behavior questions (median & mean measured in seconds) **(int)**
    - stress: measuring user stress level on a scale of 1-5 **(int)**
    - avg_ema_spent_time: time taken to answer survey (measured in seconds) **(int)**
 
- Specific Phone Usage Data: https://github.com/DMahjoob/CAIS_DigitalWellbeing/blob/3c8106cd57b7293e0ab335e93116449857c8c562/Dataset/Sensing/Data%20Dictionary%20(Daily).csv
