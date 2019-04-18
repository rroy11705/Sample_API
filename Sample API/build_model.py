from sklearn.model_selection import train_test_split
import pandas as pd 
from model import KNeighbor_Model

def build_model():

    with open("Lib/train.csv") as f:
        df = pd.read_csv(f, sep = ',')

    df.set_index('PassengerId', inplace = True)
    print(df.info())
    df.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
    print(df.head())

    df1 = pd.DataFrame(pd.get_dummies(df[['Sex', 'Embarked']]))
    df = pd.concat([df, df1], axis = 1, join = 'inner')
    df.drop(['Sex', 'Embarked'], axis = 1, inplace = True)

    y = df['Survived']
    X=df.drop(['Survived'],axis=1)
    y.head()

    X = X.interpolate(axis = 0)
    print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    model =  KNeighbor_Model()

    model.train(X_train, y_train)
    score = model.score(X_test, y_test) 
    print(score)

    path = 'Lib/Models/Survival.pkl'
    print("Training Complete")
    model.pickle_clf(path)

if __name__ == "__main__":
    build_model()