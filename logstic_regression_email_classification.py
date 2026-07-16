import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
data={
    
     "text": [
        "free prize win now",
        "claim your free offer",
        "urgent account verify now",
        "hello how are you",
        "meeting schedule for tomorrow",
        "let us catch up today",
        "win money prize",
        "call me when free",
        "exclusive offer just for you",
        "project report attached",
        "congratulations you have won",
        "important update regarding your account",
        "free vacation offer",
        "limited time offer just for you",
        "urgent action required",
        "hello friend how are you", 
        "schedule a meeting for tomorrow",
        "let us catch up today",
        "win money prize",
        "call me when free",
        "exclusive offer just for you",
        "project report attached",
        "congratulations you have won",
        "important update regarding your account",
        "free vacation offer",
        "limited time offer just for you",
    ],
     "label": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
}

df=pd.DataFrame(data)

X=df["text"]
y=df["label"]

#train test spli
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# covert text into numeric
vectorizer=TfidfVectorizer(stop_words='english')
X_train_vc=vectorizer.fit_transform(X_train)
X_test_vc=vectorizer.transform(X_test)

# model training
model=LogisticRegression(max_iter=1000)
model.fit(X_train_vc,y_train)

# model predict
y_pred=model.predict(X_test_vc)

# accuracy score
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy is :{accuracy}")

# confusion matrix
cm=confusion_matrix(y_test,y_pred)
print(f"Confusion matrix is:{cm}")

# classification report
print("Classification report is:",classification_report(y_test,y_pred))

# User Input Prediction

#input_email=["free offer just for you"]
input_email=["win money prize"]
input_email_text=vectorizer.transform(input_email)
prediction=model.predict(input_email_text)[0]
if prediction==1:
    print("Spam")
    
else:
    print("Ham")