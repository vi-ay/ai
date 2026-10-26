# Practical Exam Reference: AI/ML and Jira

Quick-lookup guide with every question, its answer and (for AI/ML) tested Python code.

- **Module 1:** AI and Machine Learning (29 questions)
- **Module 2:** Jira (34 practicals)

> **Tip:** Press `Ctrl+F` (or use the browser's find) and type `Q17` or `P21` to jump straight to a question.

## Table of Contents

- [Module 1: AI and Machine Learning](#module-1-ai-and-machine-learning)
- [Module 2: Jira](#module-2-jira)

### Module 1 questions

- [Q1. Vacuum-cleaner PEAS and table-driven agent](#q1-vacuum-cleaner-peas-and-table-driven-agent)
- [Q2. Autonomous taxi PEAS](#q2-autonomous-taxi-peas)
- [Q3. 8-puzzle BFS shortest path](#q3-8-puzzle-bfs-shortest-path)
- [Q4. Water-jug DFS (4L and 3L, target 2L)](#q4-water-jug-dfs-4l-and-3l-target-2l)
- [Q5. Linear Regression](#q5-linear-regression)
- [Q6. Decision Tree classifier on Iris](#q6-decision-tree-classifier-on-iris)
- [Q7. Decision Tree Regression](#q7-decision-tree-regression)
- [Q8. kNN classifier](#q8-knn-classifier)
- [Q9. Multiclass Logistic Regression](#q9-multiclass-logistic-regression)
- [Q10. Class imbalance and bias analysis](#q10-class-imbalance-and-bias-analysis)
- [Q11. Naive Bayes with train/test split](#q11-naive-bayes-with-traintest-split)
- [Q12. ML workflow with Pandas](#q12-ml-workflow-with-pandas)
- [Q13. 8-puzzle BFS move sequence](#q13-8-puzzle-bfs-move-sequence)
- [Q14. Water-jug DFS (4L and 3L)](#q14-water-jug-dfs-4l-and-3l)
- [Q15. Bayes Rule disease probability](#q15-bayes-rule-disease-probability)
- [Q16. Naive Bayes weather prediction](#q16-naive-bayes-weather-prediction)
- [Q17. MNIST-style binary classifier](#q17-mnist-style-binary-classifier)
- [Q18. kNN accuracy](#q18-knn-accuracy)
- [Q19. Multiclass classification](#q19-multiclass-classification)
- [Q20. Class imbalance counts](#q20-class-imbalance-counts)
- [Q21. Water-jug DFS custom capacities](#q21-water-jug-dfs-custom-capacities)
- [Q22. Naive Bayes and dataset bias](#q22-naive-bayes-and-dataset-bias)
- [Q23. Decision Tree with bias check](#q23-decision-tree-with-bias-check)
- [Q24. Basic ML workflow](#q24-basic-ml-workflow)
- [Q25. Linear Regression evaluation](#q25-linear-regression-evaluation)
- [Q26. Decision Tree classifier on Iris](#q26-decision-tree-classifier-on-iris)
- [Q27. Decision Tree regression evaluation](#q27-decision-tree-regression-evaluation)
- [Q28. Class distribution and bias](#q28-class-distribution-and-bias)
- [Q29. Simple table-driven agent](#q29-simple-table-driven-agent)

### Module 2 practicals

- [P1. Kanban project, details, dashboard](#p1-kanban-project-details-dashboard)
- [P2. Add team members and roles](#p2-add-team-members-and-roles)
- [P3. Issue types, components, versions](#p3-issue-types-components-versions)
- [P4. Epic, issues, assignee, priority, due date](#p4-epic-issues-assignee-priority-due-date)
- [P5. Automation rule (trigger, condition, action)](#p5-automation-rule-trigger-condition-action)
- [P6. Jira mobile app](#p6-jira-mobile-app)
- [P7. Scrum project, sprint, review](#p7-scrum-project-sprint-review)
- [P8. Issue types, components, versions](#p8-issue-types-components-versions)
- [P9. Epic with Stories and Tasks](#p9-epic-with-stories-and-tasks)
- [P10. Custom workflow and association](#p10-custom-workflow-and-association)
- [P11. Pie chart report on dashboard](#p11-pie-chart-report-on-dashboard)
- [P12. Create, test, enable automation](#p12-create-test-enable-automation)
- [P13. Project roles and permissions review](#p13-project-roles-and-permissions-review)
- [P14. End-to-end Scrum project and Sprint Report](#p14-end-to-end-scrum-project-and-sprint-report)
- [P15. Workflow, JQL, dashboard, automation](#p15-workflow-jql-dashboard-automation)
- [P16. Account, dashboard, project setup](#p16-account-dashboard-project-setup)
- [P17. Team members and roles](#p17-team-members-and-roles)
- [P18. Issue types, components, versions](#p18-issue-types-components-versions)
- [P19. Mobile app status, comment, assignee](#p19-mobile-app-status-comment-assignee)
- [P20. Scrum project with Epic, 2 Stories, 1 Task](#p20-scrum-project-with-epic-2-stories-1-task)
- [P21. Create and save JQL filter](#p21-create-and-save-jql-filter)
- [P22. Dashboard with issue statistics gadgets](#p22-dashboard-with-issue-statistics-gadgets)
- [P23. Roles and permission access](#p23-roles-and-permission-access)
- [P24. Automation: assign on creation or transition](#p24-automation-assign-on-creation-or-transition)
- [P25. Scrum board, sprint, report](#p25-scrum-board-sprint-report)
- [P26. Link issues to components and versions](#p26-link-issues-to-components-and-versions)
- [P27. Filters and issue statistics summary](#p27-filters-and-issue-statistics-summary)
- [P28. Mobile app with notifications](#p28-mobile-app-with-notifications)
- [P29. Account, project, details (A, B, C)](#p29-account-project-details-a-b-c)
- [P30. Add members and assign roles (A, B)](#p30-add-members-and-assign-roles-a-b)
- [P31. Custom workflow, conditions, publish (A, B, C)](#p31-custom-workflow-conditions-publish-a-b-c)
- [P32. Scrum board, sprint lifecycle, report (A, B, C)](#p32-scrum-board-sprint-lifecycle-report-a-b-c)
- [P33. Burndown, velocity, pie chart, dashboard (A, B, C)](#p33-burndown-velocity-pie-chart-dashboard-a-b-c)
- [P34. Permission scheme and roles (A, B)](#p34-permission-scheme-and-roles-a-b)

---

## Module 1: AI and Machine Learning

**Requirements:** Python 3, pandas, scikit-learn. Install with `pip install pandas scikit-learn numpy`.

**Notes:**

- The 8-puzzle examples use sample start and goal states. Replace them with the exact configurations from your question paper.
- `fetch_california_housing()` needs internet the first time. Offline fallback: `load_diabetes()`.
- Q17 uses scikit-learn's small 8x8 `load_digits` as a light MNIST-like dataset, not the full 28x28 MNIST.

### Quick revision table

| Question numbers | Main topic |
| --- | --- |
| 1, 29 | Table-driven agent |
| 2 | PEAS for autonomous taxi |
| 3, 13 | 8-puzzle BFS |
| 4, 14, 21 | Water-jug DFS |
| 5, 25 | Linear Regression |
| 6, 26 | Decision Tree classification |
| 7, 27 | Decision Tree regression |
| 8, 18 | kNN |
| 9, 19 | Multiclass Logistic Regression |
| 10, 20, 28 | Class imbalance and bias |
| 11, 16, 22 | Naive Bayes |
| 12, 24 | Machine-learning workflow |
| 15 | Bayes' Rule |
| 17 | Binary digit classification |

### Practical exam notes

- `X` = input features; `y` = target/output.
- `fit()` trains a model; `predict()` produces predictions; `score()` evaluates a model.
- BFS finds a shortest path when each move has equal cost. DFS does not guarantee a shortest path.
- Classification: accuracy = fraction of correct predictions.
- Regression: Mean Squared Error = average squared prediction error.
- Fit scaling on training data only, then transform both training and testing data.
- Replace example puzzle configurations with the exact initial and goal configurations from your question paper.

---

### Q1. Vacuum-cleaner PEAS and table-driven agent

> **Question:** Define PEAS for a vacuum-cleaner agent. Classify its environment as fully/partially observable, deterministic/stochastic, and episodic/sequential. Implement a simple table-driven agent in Python.

**Answer:** PEAS: Performance—clean floor, low time and energy; Environment—rooms A and B; Actuators—move left/right and suck; Sensors—location and dirt status. In this simple two-room model the environment is fully observable, deterministic and sequential (also static and discrete).

**Python program:**

```python
table = {
    ("A", "Dirty"): "Suck",
    ("A", "Clean"): "Move Right",
    ("B", "Dirty"): "Suck",
    ("B", "Clean"): "Move Left"
}

location = input("Enter location (A/B): ")
status = input("Enter status (Dirty/Clean): ")
print("Action:", table[(location, status)])
```

**Sample output / result:** Input A and Dirty gives Action: Suck.

**Answer:** The table maps each percept to an action.


[Back to top](#table-of-contents)

---

### Q2. Autonomous taxi PEAS

> **Question:** Define PEAS for an autonomous taxi and classify its task environment using suitable examples.

**Answer:** Performance: safety, reaching destination, passenger comfort, low time and cost. Environment: roads, traffic, pedestrians, passengers and signals. Actuators: steering, brakes, accelerator, horn and indicators. Sensors: cameras, GPS, radar, LiDAR and speedometer.

| Property | Classification | Example |
| --- | --- | --- |
| Observability | Partially observable | A pedestrian is hidden behind a bus. |
| Determinism | Stochastic | Another vehicle changes lanes unexpectedly. |
| Episodes | Sequential | Braking affects future movement. |
| Dynamics | Dynamic | Traffic changes while driving. |
| State | Continuous | Speed and position vary continuously. |
| Agents | Multi-agent | Other drivers and pedestrians interact. |

**Answer:** An autonomous taxi operates in a complex, partially observable, stochastic and dynamic environment.


[Back to top](#table-of-contents)

---

### Q3. 8-puzzle BFS shortest path

> **Question:** Implement Breadth-First Search (BFS) in Python to find the shortest path in an 8-puzzle. Use the given initial and goal configurations.

**Answer:** BFS explores states level by level and finds the shortest path when each move has equal cost. The example below uses 0 for the blank; replace start and goal with the configurations supplied by your instructor.

**Python program:**

```python
from collections import deque

start = (1, 2, 3, 4, 5, 6, 0, 7, 8)  # 0 is blank
goal  = (1, 2, 3, 4, 5, 6, 7, 8, 0)

queue = deque([(start, [])])
visited = {start}

while queue:
    state, path = queue.popleft()
    if state == goal:
        print("Moves:", path)
        print("Total moves:", len(path))
        break

    z = state.index(0)
    row, col = divmod(z, 3)
    options = [
        ("Up", z-3, row > 0),
        ("Down", z+3, row < 2),
        ("Left", z-1, col > 0),
        ("Right", z+1, col < 2)
    ]

    for name, pos, valid in options:
        if valid:
            new = list(state)
            new[z], new[pos] = new[pos], new[z]
            new = tuple(new)
            if new not in visited:
                visited.add(new)
                queue.append((new, path + [name]))
else:
    print("No solution found")
```

**Sample output / result:** For the example start state, moves are ['Right', 'Right']; total moves: 2.


[Back to top](#table-of-contents)

---

### Q4. Water-jug DFS (4L and 3L, target 2L)

> **Question:** Solve the 4-litre and 3-litre water-jug problem using Depth-First Search (DFS) to obtain exactly 2 litres. Display the sequence of states.

**Answer:** Represent a state as (amount in 4-litre jug, amount in 3-litre jug). DFS explores fill, empty and pour actions until one jug contains exactly 2 litres.

**Python program:**

```python
def dfs(a, b, path, visited):
    state = (a, b)
    if state in visited:
        return None
    visited.add(state)
    path = path + [state]

    if a == 2 or b == 2:
        return path

    x = min(a, 3-b)  # pour 4L jug into 3L jug
    y = min(b, 4-a)  # pour 3L jug into 4L jug
    next_states = [
        (4, b), (a, 3), (0, b), (a, 0),
        (a-x, b+x), (a+y, b-y)
    ]

    for na, nb in next_states:
        if (na, nb) != state:
            result = dfs(na, nb, path, visited)
            if result:
                return result
    return None

solution = dfs(0, 0, [], set())
print("Sequence of states:")
for state in solution:
    print(state)
```

**Sample output / result:** The sequence ends at a state where either jug contains 2 litres, for example (0, 2).

Actual output when run: the sequence ends at `(4, 2)`, so the 3L jug holds 2 litres.


[Back to top](#table-of-contents)

---

### Q5. Linear Regression

> **Question:** Implement Linear Regression using scikit-learn on a suitable dataset. Display model predictions and an evaluation metric.

**Answer:** The California Housing dataset is used to predict a continuous target. The first download may require internet access.

**Python program:**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual values:", y_test[:5])
print("Predictions:", predictions[:5])
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
```

**Sample output / result:** The program prints five actual values, five predictions and Mean Squared Error (MSE).

**Answer:** A lower MSE generally indicates smaller prediction errors on the same dataset.

> **Offline tip:** `fetch_california_housing()` downloads data the first time. If there is no internet, replace it with `from sklearn.datasets import load_diabetes` and `data = load_diabetes()`. Everything else stays the same.


[Back to top](#table-of-contents)

---

### Q6. Decision Tree classifier on Iris

> **Question:** Implement a Decision Tree classifier using the Iris dataset. Display predictions and classification accuracy.

**Answer:** Train a Decision Tree classifier on Iris flower measurements and labels.

**Python program:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions[:10])
print("Accuracy:", accuracy_score(y_test, predictions))
```

**Sample output / result:** The program displays predicted class labels and accuracy (between 0 and 1).


[Back to top](#table-of-contents)

---

### Q7. Decision Tree Regression

> **Question:** Implement Decision Tree Regression on a suitable regression dataset and display predictions with an evaluation metric.

**Answer:** Use DecisionTreeRegressor on the California Housing dataset.

**Python program:**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions[:5])
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", model.score(X_test, y_test))
```

**Sample output / result:** The program displays predictions, Mean Squared Error and R² score.

> **Offline tip:** `fetch_california_housing()` downloads data the first time. If there is no internet, replace it with `from sklearn.datasets import load_diabetes` and `data = load_diabetes()`. Everything else stays the same.


[Back to top](#table-of-contents)

---

### Q8. kNN classifier

> **Question:** Implement a k-Nearest Neighbors (kNN) classifier using scikit-learn. Split the dataset and display test accuracy.

**Answer:** Use the Iris dataset and k=3 nearest neighbours.

**Python program:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Predictions:", model.predict(X_test[:10]))
print("Test Accuracy:", model.score(X_test, y_test))
```

**Sample output / result:** The program displays predicted class labels and test accuracy.


[Back to top](#table-of-contents)

---

### Q9. Multiclass Logistic Regression

> **Question:** Perform multiclass classification using Logistic Regression or a Decision Tree on a suitable dataset. Display the accuracy.

**Answer:** Use Logistic Regression to classify Iris flowers into three species.

**Python program:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

print("Predictions:", model.predict(X_test[:5]))
print("Accuracy:", model.score(X_test, y_test))
```

**Sample output / result:** The program displays predictions and accuracy.


[Back to top](#table-of-contents)

---

### Q10. Class imbalance and bias analysis

> **Question:** Analyze a public dataset for class imbalance and possible bias. Display class counts and briefly discuss limitations in the data.

**Answer:** Use the public Iris dataset and count samples in each class.

**Python program:**

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
print(df["Species"].value_counts().sort_index())
```

**Sample output / result:** Class 0: 50, Class 1: 50, Class 2: 50.

**Answer:** Iris is balanced by class count, but it is small and contains only three species. Balanced counts do not rule out other forms of bias.


[Back to top](#table-of-contents)

---

### Q11. Naive Bayes with train/test split

> **Question:** Implement Naïve Bayes classification using a small dataset, and show how the dataset is divided into training and testing sets.

**Answer:** Encode a small weather dataset, split it into training and testing sets, and train Categorical Naïve Bayes.

**Python program:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB

weather = [
    ["Sunny"], ["Sunny"], ["Rainy"], ["Rainy"], ["Overcast"],
    ["Overcast"], ["Sunny"], ["Rainy"], ["Overcast"], ["Sunny"]
]
play = ["No", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No"]

enc = OrdinalEncoder()
X = enc.fit_transform(weather)
le = LabelEncoder()
y = le.fit_transform(play)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = CategoricalNB()
model.fit(X_train, y_train)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
test = enc.transform([["Overcast"]])
print("Prediction:", le.inverse_transform(model.predict(test))[0])
```

**Sample output / result:** Training samples: 7; Testing samples: 3; a prediction is printed.


[Back to top](#table-of-contents)

---

### Q12. ML workflow with Pandas

> **Question:** Implement a machine-learning workflow using a suitable dataset: load with Pandas, handle missing values or scale features, split into training and testing sets, train a classifier, and display training and testing accuracy.

**Answer:** This workflow loads Iris with Pandas, fills missing values if present, scales features, splits the data and trains a Decision Tree.

**Python program:**

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
df = df.fillna(df.mean())  # Iris has no missing values

X = df.drop("Species", axis=1)
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))
```

**Sample output / result:** Training Accuracy and Testing Accuracy are printed. Iris has no missing values, so fillna does not change it.


[Back to top](#table-of-contents)

---

### Q13. 8-puzzle BFS move sequence

> **Question:** Write a Python program to solve the 8-puzzle using Breadth-First Search (BFS) and display the sequence of moves.

**Answer:** This is the same BFS approach as Question 3. The program prints the move sequence and its length.

**Python program:**

```python
from collections import deque

start = (1, 2, 3, 4, 5, 6, 0, 7, 8)  # 0 is blank
goal  = (1, 2, 3, 4, 5, 6, 7, 8, 0)

queue = deque([(start, [])])
visited = {start}

while queue:
    state, path = queue.popleft()
    if state == goal:
        print("Moves:", path)
        print("Total moves:", len(path))
        break

    z = state.index(0)
    row, col = divmod(z, 3)
    options = [
        ("Up", z-3, row > 0),
        ("Down", z+3, row < 2),
        ("Left", z-1, col > 0),
        ("Right", z+1, col < 2)
    ]

    for name, pos, valid in options:
        if valid:
            new = list(state)
            new[z], new[pos] = new[pos], new[z]
            new = tuple(new)
            if new not in visited:
                visited.add(new)
                queue.append((new, path + [name]))
else:
    print("No solution found")
```

**Sample output / result:** For the sample state: ['Right', 'Right']; total moves: 2.


[Back to top](#table-of-contents)

---

### Q14. Water-jug DFS (4L and 3L)

> **Question:** Use Depth-First Search (DFS) in Python to solve the 4-litre and 3-litre waterjug problem and obtain exactly 2 litres

**Answer:** This uses the same state representation and DFS method as Question 4.

**Python program:**

```python
def dfs(a, b, path, visited):
    state = (a, b)
    if state in visited:
        return None
    visited.add(state)
    path = path + [state]

    if a == 2 or b == 2:
        return path

    x = min(a, 3-b)  # pour 4L jug into 3L jug
    y = min(b, 4-a)  # pour 3L jug into 4L jug
    next_states = [
        (4, b), (a, 3), (0, b), (a, 0),
        (a-x, b+x), (a+y, b-y)
    ]

    for na, nb in next_states:
        if (na, nb) != state:
            result = dfs(na, nb, path, visited)
            if result:
                return result
    return None

solution = dfs(0, 0, [], set())
print("Sequence of states:")
for state in solution:
    print(state)
```

**Sample output / result:** The state sequence reaches a state where either jug contains exactly 2 litres.

Actual output when run: the sequence ends at `(4, 2)`, so the 3L jug holds 2 litres.


[Back to top](#table-of-contents)

---

### Q15. Bayes Rule disease probability

> **Question:** Write a Python program applying Bayes’ Rule to calculate the probability of disease after a positive test result using the given scenario.

**Answer:** Example scenario: disease prevalence 1%, sensitivity 99%, and false-positive rate 5%. Bayes' Rule combines these values to find the probability of disease given a positive result.

**Python program:**

```python
prevalence = 0.01
sensitivity = 0.99
false_positive_rate = 0.05

p_positive = (
    sensitivity * prevalence
    + false_positive_rate * (1 - prevalence)
)
p_disease_given_positive = sensitivity * prevalence / p_positive

print("Probability of disease after positive test:",
      round(p_disease_given_positive * 100, 2), "%")
```

**Sample output / result:** Probability of disease after positive test: 16.67%.


[Back to top](#table-of-contents)

---

### Q16. Naive Bayes weather prediction

> **Question:** Implement a Naïve Bayes classifier using scikit-learn on a small weather or spam dataset and display its prediction.

**Answer:** Use the small weather dataset and CategoricalNB to predict whether to play when the weather is Overcast.

**Python program:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB

weather = [
    ["Sunny"], ["Sunny"], ["Rainy"], ["Rainy"], ["Overcast"],
    ["Overcast"], ["Sunny"], ["Rainy"], ["Overcast"], ["Sunny"]
]
play = ["No", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No"]

enc = OrdinalEncoder()
X = enc.fit_transform(weather)
le = LabelEncoder()
y = le.fit_transform(play)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = CategoricalNB()
model.fit(X_train, y_train)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
test = enc.transform([["Overcast"]])
print("Prediction:", le.inverse_transform(model.predict(test))[0])
```

**Sample output / result:** The program prints the training/testing sizes and a prediction for Overcast.


[Back to top](#table-of-contents)

---

### Q17. MNIST-style binary classifier

> **Question:** Use the MNIST dataset to build a binary classifier for digit versus not-digit and evaluate its performance

**Answer:** For a lightweight practical, use scikit-learn's load_digits dataset (8×8 images) as an MNIST-like alternative. The binary target is digit 5 versus not 5.

**Python program:**

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

digits = load_digits()  # Small 8x8 dataset, MNIST-like
X = digits.data
y = (digits.target == 5).astype(int)  # 1 = digit 5, 0 = not 5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions (1=5, 0=not 5):", predictions[:10])
print("Accuracy:", accuracy_score(y_test, predictions))
```

**Sample output / result:** The program displays predictions (1 means digit 5; 0 means not 5) and accuracy.

**Answer:** This is not the full 28×28 MNIST dataset; use a downloaded MNIST dataset if your instructor requires MNIST specifically.

Actual output when run: accuracy is about `0.9956`.


[Back to top](#table-of-contents)

---

### Q18. kNN accuracy

> **Question:** Implement a k-Nearest Neighbors (kNN) classifier using a suitable dataset and evaluate its accuracy.

**Answer:** Use the Iris dataset with KNeighborsClassifier and k=3.

**Python program:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Predictions:", model.predict(X_test[:10]))
print("Test Accuracy:", model.score(X_test, y_test))
```

**Sample output / result:** The program displays predictions and test accuracy.


[Back to top](#table-of-contents)

---

### Q19. Multiclass classification

> **Question:** Perform multiclass classification using Logistic Regression or a Decision Tree on a suitable dataset.

**Answer:** Use Logistic Regression to predict the three Iris species.

**Python program:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

print("Predictions:", model.predict(X_test[:5]))
print("Accuracy:", model.score(X_test, y_test))
```

**Sample output / result:** The program prints five predictions and the model accuracy.


[Back to top](#table-of-contents)

---

### Q20. Class imbalance counts

> **Question:** Select a public dataset and examine possible class imbalance by displaying the number of samples in each class.

**Answer:** Count the class labels in the Iris dataset.

**Python program:**

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
print(df["Species"].value_counts().sort_index())
```

**Sample output / result:** Each of the three classes contains 50 samples.

**Answer:** The dataset has equal class counts.


[Back to top](#table-of-contents)

---

### Q21. Water-jug DFS custom capacities

> **Question:** Write a Python program to solve a water-jug problem using DFS; use jug capacities of your choice and reach a specified target amount.

**Answer:** Use a 5-litre jug and a 3-litre jug to obtain exactly 4 litres in the first jug.

**Python program:**

```python
def dfs(a, b, path, visited):
    state = (a, b)
    if state in visited:
        return None
    visited.add(state)
    path = path + [state]
    if a == 4:
        return path

    x = min(a, 3-b)
    y = min(b, 5-a)
    next_states = [
        (5, b), (a, 3), (0, b), (a, 0),
        (a-x, b+x), (a+y, b-y)
    ]
    for na, nb in next_states:
        if (na, nb) != state:
            result = dfs(na, nb, path, visited)
            if result:
                return result
    return None

solution = dfs(0, 0, [], set())
for state in solution:
    print(state)
```

**Sample output / result:** The program prints a sequence reaching a state where the 5-litre jug contains 4 litres, such as (4, 0).


[Back to top](#table-of-contents)

---

### Q22. Naive Bayes and dataset bias

> **Question:** Train a Naïve Bayes classifier on a small dataset, then discuss one possible source of bias or class imbalance in that dataset.

**Answer:** Train Naïve Bayes on the small weather dataset. A possible limitation is that a tiny dataset may not represent real weather conditions; if the Yes/No classes are imbalanced, the model may favour the majority class.

**Python program:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB

weather = [
    ["Sunny"], ["Sunny"], ["Rainy"], ["Rainy"], ["Overcast"],
    ["Overcast"], ["Sunny"], ["Rainy"], ["Overcast"], ["Sunny"]
]
play = ["No", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No"]

enc = OrdinalEncoder()
X = enc.fit_transform(weather)
le = LabelEncoder()
y = le.fit_transform(play)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = CategoricalNB()
model.fit(X_train, y_train)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
test = enc.transform([["Overcast"]])
print("Prediction:", le.inverse_transform(model.predict(test))[0])
```

**Sample output / result:** The program prints training/testing sample counts and a prediction.


[Back to top](#table-of-contents)

---

### Q23. Decision Tree with bias check

> **Question:** Build a Decision Tree classifier and assess possible bias and class imbalance in the selected dataset.

**Answer:** Use Iris, display class counts, and compare training and testing accuracy.

**Python program:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target
print("Class counts:")
print(pd.Series(y).value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))
```

**Sample output / result:** Iris has equal class counts. A deep tree can overfit; equal class counts do not eliminate all data bias.


[Back to top](#table-of-contents)

---

### Q24. Basic ML workflow

> **Question:** Demonstrate a basic machine-learning workflow: load a dataset with Pandas, preprocess it, split it, train a classifier and report training/testing accuracy.

**Answer:** Load Iris into a DataFrame, fill missing values if any, scale the features, split the dataset, train a Decision Tree and report accuracy.

**Python program:**

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
df = df.fillna(df.mean())  # Iris has no missing values

X = df.drop("Species", axis=1)
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))
```

**Sample output / result:** Training and testing accuracy are printed.


[Back to top](#table-of-contents)

---

### Q25. Linear Regression evaluation

> **Question:** Implement Linear Regression using scikit-learn on a suitable regression dataset and evaluate the model.

**Answer:** Use California Housing and evaluate predictions with Mean Squared Error.

**Python program:**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual values:", y_test[:5])
print("Predictions:", predictions[:5])
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
```

**Sample output / result:** The program displays actual values, predictions and MSE.

> **Offline tip:** `fetch_california_housing()` downloads data the first time. If there is no internet, replace it with `from sklearn.datasets import load_diabetes` and `data = load_diabetes()`. Everything else stays the same.


[Back to top](#table-of-contents)

---

### Q26. Decision Tree classifier on Iris

> **Question:** Implement a Decision Tree classifier using the Iris dataset or another suitable dataset

**Answer:** Use the Iris dataset and DecisionTreeClassifier.

**Python program:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions[:10])
print("Accuracy:", accuracy_score(y_test, predictions))
```

**Sample output / result:** The program displays predictions and classification accuracy.


[Back to top](#table-of-contents)

---

### Q27. Decision Tree regression evaluation

> **Question:** Implement Decision Tree regression on a suitable regression dataset and evaluate its predictions

**Answer:** Use California Housing and DecisionTreeRegressor.

**Python program:**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions[:5])
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", model.score(X_test, y_test))
```

**Sample output / result:** The program displays predicted values, MSE and R² score.

> **Offline tip:** `fetch_california_housing()` downloads data the first time. If there is no internet, replace it with `from sklearn.datasets import load_diabetes` and `data = load_diabetes()`. Everything else stays the same.


[Back to top](#table-of-contents)

---

### Q28. Class distribution and bias

> **Question:** Select a public dataset and analyze possible bias and class imbalance. Report the class distribution and observations

**Answer:** Use the Iris dataset to count samples in each species.

**Python program:**

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
print(df["Species"].value_counts().sort_index())
```

**Sample output / result:** Class 0: 50, Class 1: 50, Class 2: 50.

**Answer:** The dataset is balanced by class count, but its small size and limited species coverage are limitations.


[Back to top](#table-of-contents)

---

### Q29. Simple table-driven agent

> **Question:** Create a simple table-driven agent in Python. Define its percepts, actions and table, then test it with sample inputs.

**Answer:** Percepts: location and room status. Actions: Suck, Move Right and Move Left. The table maps each percept to an action.

**Python program:**

```python
table = {
    ("A", "Dirty"): "Suck",
    ("A", "Clean"): "Move Right",
    ("B", "Dirty"): "Suck",
    ("B", "Clean"): "Move Left"
}

percepts = [
    ("A", "Dirty"), ("A", "Clean"),
    ("B", "Dirty"), ("B", "Clean")
]
for percept in percepts:
    print("Percept:", percept, "Action:", table[percept])
```

**Sample output / result:** ('A', 'Dirty') -> Suck; ('A', 'Clean') -> Move Right; ('B', 'Dirty') -> Suck; ('B', 'Clean') -> Move Left.


[Back to top](#table-of-contents)

---

## Module 2: Jira

All steps are written out in full for every practical (no "same as" shortcuts).

### P1. Kanban project, details, dashboard

> **Question:** Create a Jira Cloud project (Kanban), configure project details, and explore the project dashboard.

**Step 1: Create a Kanban project**

1. Log in to your Atlassian Jira Cloud account.
2. In the top navigation bar, click **Projects** (or **Spaces**), then select **Create project**.
3. Choose the **Kanban** project template.
4. Select **Company-managed project** (or Team-managed, depending on preference).
5. Enter the project name (e.g., *Software Development Kanban*).
6. Verify the auto-generated **Key** (e.g., SDK).
7. Click **Create project**.

**Step 2: Configure project details**

1. In the project left sidebar, click **Project settings** (or Space settings).
2. Select **Details**.
3. Update the **Project Name**, **Project Category** and enter a short **Description**.
4. Set a **Project Lead** from the dropdown.
5. Click **Save changes**.

**Step 3: Explore the project dashboard**

1. Click **Summary** or **Kanban board** in the left sidebar.
2. Observe the default Kanban columns (To Do, In Progress, Done).
3. Check the top menu for Filters, Dashboards, People and Apps.


[Back to top](#table-of-contents)

---

### P2. Add team members and roles

> **Question:** Create a Jira project, add team members, and assign appropriate project roles

**Step 1: Add team members**

1. Open your project.
2. Click **Add people** (+ icon) at the top right of the board, or go to **Project settings > People**.
3. Click **Add people**.
4. Enter the email addresses of the team members.

**Step 2: Assign project roles**

1. In the **Role** dropdown next to each email, select the role (Developer, Member or Administrator).
2. Click **Add**.
3. To change a role later: **Project settings > People**, find the user, click the current role dropdown and choose a new role (e.g., Administrator).


[Back to top](#table-of-contents)

---

### P3. Issue types, components, versions

> **Question:** Configure issue types (Epic, Story, Task, Bug), and create project components and versions.

**Step 1: Configure issue types**

1. Go to **Project settings > Issue types**.
2. Review the existing issue types: **Epic, Story, Task, Bug**.
3. Customize the fields of an issue type: drag required fields from the right panel into the issue form layout.
4. Click **Save changes**.

**Step 2: Create project components**

1. In the left sidebar, click **Components** (or Project settings > Components).
2. Click **Create component**.
3. Enter the Component Name (e.g., *Database*).
4. Add a Description (e.g., *Database related issues and scripts*).
5. Set the Component Lead and Default Assignee (e.g., Project Lead).
6. Click **Save**.

**Step 3: Create project versions (releases)**

1. Click **Releases** (or Versions) in the left sidebar.
2. Click **Create version**.
3. Enter the version Name (e.g., *Version 1.0*).
4. Select a Start date and Release date.
5. Add a Description (e.g., *Initial Major Release*).
6. Click **Save**.


[Back to top](#table-of-contents)

---

### P4. Epic, issues, assignee, priority, due date

> **Question:** Create an Epic and add related issues. Assign issues to users and set priority and due dates.

**Step 1: Create an Epic**

1. Click **+ Create** in the top navigation bar.
2. Set **Issue Type** to **Epic**.
3. Enter the Summary (e.g., *User Authentication Module*).
4. Click **Create**.

**Step 2: Add related issues (child work items)**

1. Open the created Epic.
2. In **Child work items**, click **+ Add child item**.
3. Select **Story** or **Task**, enter the title (e.g., *Create Login Page*) and press **Enter**.

**Step 3: Assign, set priority and due date**

1. Click the child story to open its details.
2. Click **Assignee** and select a user, or click **Assign to me**.
3. Click **Priority** and choose a level (e.g., High).
4. Click **Due date** and pick a date from the calendar.


[Back to top](#table-of-contents)

---

### P5. Automation rule (trigger, condition, action)

> **Question:** Create a Jira Automation rule with a trigger, condition, and action—for example, when an issue moves to Done, send a notification or auto-assign it.

**Step 1: Open Jira Automation**

1. Go to **Project settings > Automation**.
2. Click **Create rule**.

**Step 2: Set the trigger**

1. Search for and select **Issue transitioned**.
2. Set **From status** = Any status and **To status** = Done.
3. Click **Save**.

**Step 3: Set the condition**

1. Click **Add component > New condition**.
2. Select **Issue fields condition**.
3. Set Field = Issue Type, Condition = equals, Value = Bug.
4. Click **Save**.

**Step 4: Set the action**

1. Click **Add component > New action**.
2. Select **Send email** or **Assign issue**.
3. Choose Assignee > **User who triggered the event** (or select a Lead).
4. Click **Save**.
5. Enter a Rule Name (e.g., *Auto Assign Bug on Done*) and click **Turn it on**.


[Back to top](#table-of-contents)

---

### P6. Jira mobile app

> **Question:** Configure the Jira mobile app, view an issue, update its status, add a comment, and assign it to a user.

**Step 1: Install and configure**

1. Download **Jira Cloud** from Google Play Store or Apple App Store.
2. Open the app and log in with your Atlassian account.
3. Select your workspace and project from the list.

**Step 2: View and update an issue**

1. Tap an assigned issue from the list or board view.
2. Tap the **Status** dropdown and change **To Do** to **In Progress**.
3. Tap **Add Comment**, type a status update (e.g., *Working on this task*) and tap **Send**.
4. Tap the **Assignee** field and select a teammate.


[Back to top](#table-of-contents)

---

### P7. Scrum project, sprint, review

> **Question:** Create a Scrum project and board, add issues to a sprint, start the sprint, and review sprint progress

**Step 1: Create a Scrum project and board**

1. Click **Projects > Create project**.
2. Choose the **Scrum** template, then **Company-managed project**.
3. Enter a Name (e.g., *E-Commerce Application*) and Key (e.g., ECOM).
4. Click **Create project**.

**Step 2: Add issues to a sprint**

1. Click **Backlog** in the left sidebar.
2. Click **+ Create issue** at the bottom of the Backlog to add stories/tasks.
3. Click **Create sprint** if no sprint exists, then drag items from the Backlog up into the **Sprint 1** section.

**Step 3: Start the sprint and review progress**

1. Click **Start sprint** at the top right of the sprint container.
2. Set the Sprint Name, Duration (e.g., 2 weeks) and Sprint Goal.
3. Click **Start**.
4. Open **Active sprints** (or the board) in the sidebar and review how tasks move across To Do, In Progress and Done.


[Back to top](#table-of-contents)

---

### P8. Issue types, components, versions

> **Question:** Create and configure issue types, components, and versions in a Jira project

**Step 1: Configure issue types (Stories, Bugs, Tasks, Epics)**

1. Go to **Project settings > Issue types**.
2. Review the existing issue types: **Epic, Story, Task, Bug**.
3. Customize the fields of an issue type: drag required fields from the right panel into the issue form layout.
4. Click **Save changes**.

**Step 2: Create project components**

1. In the left sidebar, click **Components** (or Project settings > Components).
2. Click **Create component**.
3. Enter the Component Name (e.g., *Database*).
4. Add a Description (e.g., *Database related issues and scripts*).
5. Set the Component Lead and Default Assignee (e.g., Project Lead).
6. Click **Save**.

**Step 3: Create project versions (releases)**

1. Click **Releases** (or Versions) in the left sidebar.
2. Click **Create version**.
3. Enter the version Name (e.g., *Version 1.0*).
4. Select a Start date and Release date.
5. Add a Description (e.g., *Initial Major Release*).
6. Click **Save**.


[Back to top](#table-of-contents)

---

### P9. Epic with Stories and Tasks

> **Question:** Create an Epic with Stories and Tasks, assign the issues, and set priority and due date.

**Step 1: Create an Epic**

1. Click **+ Create** in the top navigation bar.
2. Set **Issue Type** to **Epic**.
3. Enter the Summary (e.g., *User Authentication Module*).
4. Click **Create**.

**Step 2: Add related issues (child work items)**

1. Open the created Epic.
2. In **Child work items**, click **+ Add child item**.
3. Select **Story** or **Task**, enter the title (e.g., *Create Login Page*) and press **Enter**.

**Step 3: Assign, set priority (High/Medium) and due date**

1. Click the child story to open its details.
2. Click **Assignee** and select a user, or click **Assign to me**.
3. Click **Priority** and choose a level (e.g., High).
4. Click **Due date** and pick a date from the calendar.


[Back to top](#table-of-contents)

---

### P10. Custom workflow and association

> **Question:** Customize a Jira workflow by adding statuses and transitions, then associate it with a project.

**Step 1: Design the workflow**

1. Go to **Project settings > Workflows**.
2. Click the **edit** (pencil) icon next to the active workflow.
3. Click **Add status**. Enter the name (e.g., *In QA*) and category (In Progress), then click **Add**.

**Step 2: Add transitions**

1. Click **Add transition**.
2. Select **From status** (In Progress) and **To status** (In QA).
3. Enter the Transition Name (e.g., *Send to QA*) and click **Add**.

**Step 3: Publish and associate the workflow**

1. Click **Publish draft** in the top alert bar.
2. Save a backup if prompted, click **Next**, map statuses if required, and click **Associate**.


[Back to top](#table-of-contents)

---

### P11. Pie chart report on dashboard

> **Question:** Generate a Pie Chart or Issue Statistics report and add it to a custom Jira dashboard

**Step 1: Generate the Pie Chart report**

1. Click **Reports** in the sidebar menu.
2. Select **Pie Chart Report** under Issue Analysis.
3. Select your project/filter and set **Statistic Type** to Assignee or Priority.
4. Click **Next** to view the report.

**Step 2: Add it to a custom dashboard**

1. Click **Dashboards** in the top menu > **Create dashboard**.
2. Enter a Name (e.g., *Project Performance Dashboard*) and click **Save**.
3. Click **+ Add gadget**, search for **Pie Chart**, and click **Add gadget**.
4. Select your project, choose Statistic Type (Priority) and click **Save**.


[Back to top](#table-of-contents)

---

### P12. Create, test, enable automation

> **Question:** Create a Jira Automation rule, test it, and enable it after verifying the trigger-condition-action sequence.

**Step 1: Create the rule sequence**

1. Go to **Project settings > Automation > Create rule**.
2. **Trigger:** Field value changed (Field: Status to In Progress).
3. **Condition:** Issue fields condition (Priority equals High).
4. **Action:** Send email to Assignee with subject *High Priority Task Started*.

**Step 2: Test and enable**

1. Save and name the rule *Notify High Priority Start*. Enable the toggle switch.
2. Go to the project board, pick a High priority issue and move its status to **In Progress**.
3. Check the **Audit log** in Project settings > Automation to verify the rule ran successfully.


[Back to top](#table-of-contents)

---

### P13. Project roles and permissions review

> **Question:** Configure project roles and review the permissions granted to each role in a Jira project

**Step 1: Configure roles**

1. Go to **Project settings > People**.
2. Review assigned roles (Administrators, Developers, Viewers).
3. Click **Add people** to assign specific roles to team members.

**Step 2: Review the permission scheme**

1. Go to **Project settings > Permissions**.
2. Review actions granted per role (e.g., Browse Projects, Create Issues, Delete Issues).
3. Verify restricted actions (like Delete Issues) are limited to Administrators.


[Back to top](#table-of-contents)

---

### P14. End-to-end Scrum project and Sprint Report

> **Question:** Create and configure a Jira Scrum project. Add team members and roles, create issue types and an Epic with related issues, create and start a sprint, and generate a Sprint Report.

1. **Create project:** Projects > Create project > choose **Scrum** > Name: *Banking App*.
2. **Add members:** Project settings > People > add members as **Developers**.
3. **Issue creation:** **+ Create** > add an Epic (*Payment Gateway*), then add child Stories (*Credit Card Support*, *UPI Integration*).
4. **Sprint execution:** Backlog > move the Stories into the sprint > **Start sprint** (set 2-week duration).
5. **Generate report:** after completing the work, go to **Reports > Sprint Report** to inspect completed vs non-completed work points.


[Back to top](#table-of-contents)

---

### P15. Workflow, JQL, dashboard, automation

> **Question:** Create a Jira project and configure a custom workflow, JQL filters, a dashboard with suitable gadgets, and an automation rule. Demonstrate the configuration and test the automation

1. **Project:** create a new project named *IT Support*.
2. **Workflow:** add a status *Awaiting Feedback* in Project settings > Workflows (then publish).
3. **JQL filter:** Filters > Search issues > Switch to JQL > type `project = "IT Support" AND priority = High` > Save as *High Priority Support*.
4. **Dashboard:** create dashboard *IT Summary* > add the **Filter Results** gadget pointing to *High Priority Support*.
5. **Automation:** create rule: Trigger = **Issue Created** > Action = **Set Assignee to Project Lead**. Test by creating a new issue.


[Back to top](#table-of-contents)

---

### P16. Account, dashboard, project setup

> **Question:** Create a Jira Cloud account, explore the dashboard, create a company-managed Scrum or Kanban project, and configure its details.

**Step 1: Create the account and explore**

1. Go to atlassian.com and sign up or log in using a Google or Atlassian account.
2. Open your Jira Cloud site (the dashboard opens after login).
3. Explore the default dashboard and its metrics.
4. Check the top navigation menu: Projects, Filters, Dashboards, People, Apps.

**Step 2: Create a company-managed project**

1. Click **Projects > Create project**.
2. Choose a **Scrum** or **Kanban** template.
3. Select **Company-managed project**.
4. Enter a Name and Key, click **Create project**.

**Step 3: Configure project details**

1. Go to **Project settings > Details**.
2. Set the **Name**, **URL** and **Avatar**.
3. Click **Save changes**.


[Back to top](#table-of-contents)

---

### P17. Team members and roles

> **Question:** Add team members to a Jira project and assign suitable project roles.

**Step 1: Add team members**

1. Open your project.
2. Click **Add people** (+ icon) at the top right of the board, or go to **Project settings > People**.
3. Click **Add people**.
4. Enter the email addresses of the team members.

**Step 2: Assign project roles**

1. In the **Role** dropdown next to each email, select the role (Developer, Member or Administrator).
2. Click **Add**.
3. To change a role later: **Project settings > People**, find the user, click the current role dropdown and choose a new role (e.g., Administrator).


[Back to top](#table-of-contents)

---

### P18. Issue types, components, versions

> **Question:** Create and configure the Epic, Story, Task, and Bug issue types. Create project components and versions.

**Step 1: Configure issue types**

1. Go to **Project settings > Issue types**.
2. Review the existing issue types: **Epic, Story, Task, Bug**.
3. Customize the fields of an issue type: drag required fields from the right panel into the issue form layout.
4. Click **Save changes**.

**Step 2: Create project components**

1. In the left sidebar, click **Components** (or Project settings > Components).
2. Click **Create component**.
3. Enter the Component Name (e.g., *Database*).
4. Add a Description (e.g., *Database related issues and scripts*).
5. Set the Component Lead and Default Assignee (e.g., Project Lead).
6. Click **Save**.

**Step 3: Create project versions (releases)**

1. Click **Releases** (or Versions) in the left sidebar.
2. Click **Create version**.
3. Enter the version Name (e.g., *Version 1.0*).
4. Select a Start date and Release date.
5. Add a Description (e.g., *Initial Major Release*).
6. Click **Save**.


[Back to top](#table-of-contents)

---

### P19. Mobile app status, comment, assignee

> **Question:** Install and configure the Jira mobile app. View an issue and update its status, comment, and assignee.

1. Download **Jira Cloud** from Google Play Store or Apple App Store.
2. Open the app and log in with your Atlassian account.
3. Select the target project and open an assigned issue card.
4. Change the status dropdown to **In Progress**.
5. Add comment: *Working on this task*.
6. Reassign to another user via the **Assignee** picker.


[Back to top](#table-of-contents)

---

### P20. Scrum project with Epic, 2 Stories, 1 Task

> **Question:** Create a company-managed Scrum project and add an Epic with at least two Stories and one Task.

1. Click **Projects > Create project > Scrum > Company-managed**.
2. Project Name: *Mobile Wallet*. Key: *MW*.
3. Click **+ Create** > Issue type: **Epic** > Name: *User Onboarding*. Click **Create**.
4. Open the Epic > click **+ Add child work item** > add **Story 1** (*User Signup*).
5. Click **+ Add child work item** > add **Story 2** (*OTP Verification*).
6. Click **+ Add child work item** > add **Task 1** (*Setup SMS Gateway*).


[Back to top](#table-of-contents)

---

### P21. Create and save JQL filter

> **Question:** Create and save a JQL filter for issues with a selected priority and status, then verify the results

**Step 1: Write the JQL query**

1. Click **Filters** in the top bar > **View all filters** (or Search issues).
2. Click **Switch to JQL** (if in basic mode).
3. Enter the query:

   ```jql
   project = "MW" AND priority = High AND status = "To Do"
   ```
4. Click **Search** (magnifying glass icon).

**Step 2: Save and verify**

1. Click **Save as** above the query box.
2. Enter Filter Name: *MW High Priority ToDo*. Click **Save**.
3. Verify that the matching issues display correctly in the results list.


[Back to top](#table-of-contents)

---

### P22. Dashboard with issue statistics gadgets

> **Question:** Create a custom dashboard and add gadgets that display issue statistics.

1. Click **Dashboards** in the top menu > **Create dashboard**.
2. Enter Name: *Management Overview Dashboard* > click **Save**.
3. Click **Add gadget**.
4. Search for and add the **Issue Statistics** gadget.
5. In the gadget settings, select your project, set **Statistic Type** to **Status**, and click **Save**.
6. Click **Add gadget** again > select **Pie Chart** gadget > set Statistic Type to **Assignee** > click **Save**.


[Back to top](#table-of-contents)

---

### P23. Roles and permission access

> **Question:** Configure project roles and review how permissions control access to project activities

1. Go to **Project settings > Permissions**.
2. View the permission categories (Browse Projects, Create Issues, Administer Projects).
3. Click **Actions > Edit permissions**.
4. Find **Delete Issues** > click **Remove** for general roles > grant it only to **Project Role (Administrators)**.
5. Test with a non-admin account to confirm they cannot delete issues.


[Back to top](#table-of-contents)

---

### P24. Automation: assign on creation or transition

> **Question:** Create an automation rule that assigns an issue when it is created or moved to a specified status, then test it.

1. Go to **Project settings > Automation > Create rule**.
2. **Trigger:** select **Issue created** OR **Issue transitioned** (To: In Progress).
3. **Condition:** **Issue fields condition** > Field: Assignee > Condition: **is empty**.
4. **Action:** **Assign issue** > Assign to: **User who triggered the event** (or a specific user).
5. Name the rule *Auto Assign on Action* > click **Turn it on**.
6. Test by creating an unassigned issue and moving it to In Progress.


[Back to top](#table-of-contents)

---

### P25. Scrum board, sprint, report

> **Question:** Create a Scrum board, add a sprint, move issues into it, start the sprint, and view its report.

1. Go to the **Backlog** menu of your Scrum project.
2. Click **Create sprint**.
3. Drag issues from the Backlog into the new Sprint container.
4. Click **Start sprint**, configure the duration and click **Start**.
5. Once the sprint is active, move cards on the board to **Done**.
6. Click **Reports > Sprint Report** to examine progress metrics.


[Back to top](#table-of-contents)

---

### P26. Link issues to components and versions

> **Question:** Create project components and versions, then associate relevant issues with them.

1. Go to the **Components** menu > **Create component** *Frontend*.
2. Go to the **Releases** menu > **Create version** *v1.0*.
3. Open an issue (e.g., MW-1).
4. In the right-hand panel, click the **Components** field and select *Frontend*.
5. Click the **Fix versions** field and select *v1.0*.
6. Save and verify the fields are populated on the issue card.


[Back to top](#table-of-contents)

---

### P27. Filters and issue statistics summary

> **Question:** Use Jira filters and issue statistics to summarize issues by status, assignee, or priority.

1. Go to **Filters > Search issues**.
2. Switch to Basic or JQL mode and search:

   - By Status: `status = "In Progress"`
   - By Assignee: `assignee = currentUser()`
   - By Priority: `priority = High`
3. Click **Change views** to switch to list or detail view.
4. Export or summarize the data using **Reports > Issue Statistics**.


[Back to top](#table-of-contents)

---

### P28. Mobile app with notifications

> **Question:** Use the Jira mobile app to view and update an issue, add a comment, assign it, and check notifications.

1. Open the Jira Mobile App.
2. Tap the **Bell** icon (Notifications tab) to review recent updates and mentions.
3. Open an issue assigned to you.
4. Update the Status to **Done**.
5. Tap **Assignee** to reassign to a teammate.
6. Scroll to **Comments**, type a message, tag a user with `@Name`, and tap **Send**.


[Back to top](#table-of-contents)

---

### P29. Account, project, details (A, B, C)

> **Question:** A. Create a Jira Cloud account and explore the dashboard. B. Create a Company-managed Scrum/Kanban Project. C. Configure project details.

**A. Account setup**

1. Go to atlassian.com and sign up or log in using a Google or Atlassian account.
2. Open your Jira Cloud site (the dashboard opens after login).
3. Explore the default dashboard and its metrics.
4. Check the top navigation menu: Projects, Filters, Dashboards, People, Apps.

**B. Project creation**

1. Click **Projects > Create project**.
2. Select the **Scrum** or **Kanban** template.
3. Select **Company-managed**.

**C. Project details**

1. Go to **Project settings > Details**.
2. Modify the **Name**, **Key** and **Category**.
3. Click **Save**.


[Back to top](#table-of-contents)

---

### P30. Add members and assign roles (A, B)

> **Question:** A. Add team members. B. Assign roles.

**A. Add team members**

1. Click the **Add people** icon.
2. Input the team members' emails.
3. Send the invites.

**B. Assign roles**

1. Go to **Project settings > People**.
2. Assign roles (Administrator, Member, Viewer) to each user.


[Back to top](#table-of-contents)

---

### P31. Custom workflow, conditions, publish (A, B, C)

> **Question:** A. Design a custom workflow. B. Add transitions and conditions. C. Publish and associate workflow with a project.

**A. Design a custom workflow**

1. Go to **Project settings > Workflows**.
2. Edit the active scheme.
3. Add custom statuses (*In Review*, *Testing*).

**B. Add transitions and conditions**

1. Click **Add transition**.
2. Define the From and To statuses.
3. Click the transition line > **Add condition** (e.g., *Only Assignee can execute*).

**C. Publish and associate**

1. Click **Publish draft**.
2. Associate the workflow with the project issue types.


[Back to top](#table-of-contents)

---

### P32. Scrum board, sprint lifecycle, report (A, B, C)

> **Question:** A. Create and configure Scrum Board. B. Create and start a Sprint. C. Complete Sprint and generate Sprint Report.

**A. Create the Scrum board**

1. Navigate to your Scrum project's backlog panel.

**B. Create and start a sprint**

1. Drag backlog stories into the Sprint container.
2. Click **Start sprint**.
3. Define the goal and timeline.

**C. Complete the sprint and report**

1. Move all finished cards to **Done** on the board.
2. Click **Complete sprint**.
3. Go to **Reports > Sprint Report**.


[Back to top](#table-of-contents)

---

### P33. Burndown, velocity, pie chart, dashboard (A, B, C)

> **Question:** A. Generate Burndown Chart and Velocity Chart. B. Create Pie Chart / Issue Statistics Report. C. Create a Custom Dashboard with gadgets.

**A. Burndown and Velocity charts**

1. Go to **Reports** in the sidebar.
2. Click **Burndown Chart** to track remaining story points vs the sprint timeline.
3. Click **Velocity Chart** to analyze story point commitment vs completion across sprints.

**B. Pie Chart and Issue Statistics report**

1. Go to **Reports > Pie Chart Report**.
2. Select a filter or project > choose the statistic type (Priority or Status) > view the report.

**C. Custom dashboard**

1. Go to **Dashboards > Create dashboard**.
2. Click **Add gadget** > add **Burndown Chart**, **Pie Chart** and **Assigned to Me** gadgets.
3. Arrange and **Save**.


[Back to top](#table-of-contents)

---

### P34. Permission scheme and roles (A, B)

> **Question:** A. View and modify Project Permission Scheme. B. Configure Project Roles and assign permissions

**A. View and modify the permission scheme**

1. Go to **Project settings > Permissions**.
2. Click **Actions > Edit permissions**.
3. Find permissions such as **Close Issues** or **Delete Work Items**.
4. Grant them only to designated roles or groups.

**B. Configure project roles and assign access**

1. Go to **Project settings > People**.
2. Create or assign roles (Administrators, Developers, Viewers).
3. Verify access limits by attempting restricted actions under different user roles.


[Back to top](#table-of-contents)

---

### Quick reference: Jira practical groups

| Topic | Practicals |
| --- | --- |
| Project setup and details | 1, 16, 29 |
| Team members and roles | 2, 17, 30 |
| Issue types, components, versions | 3, 8, 18, 26 |
| Epic with child issues | 4, 9, 20 |
| Automation rules | 5, 12, 24, 15 |
| Mobile app | 6, 19, 28 |
| Scrum and sprints | 7, 14, 25, 32 |
| Workflows | 10, 31, 15 |
| Reports and dashboards | 11, 22, 33 |
| Permissions and roles | 13, 23, 34 |
| JQL filters and statistics | 21, 27, 15 |
