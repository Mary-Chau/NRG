import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


option = st.sidebar.radio('Select Page', ('Statistics','Blackbelt Probability', 'Registration Forecast'))


df = pd.read_csv('https://github.com/Mary-Chau/NRG/blob/main/StudentInfo1.csv')
df = df[['Start Date','Gender','Registration date','Active','Starting Age','Starting Grade','BlackBelt']]
df = df.dropna()

def stats():

        st.title('NRG New Joiner Analysis')
############################################################################################
#Pie chart of Gender
        sex=df.groupby('Gender').count()
        sex=sex['Start Date']
        sex.columns=['Count']
        labels=['F','M']
        color=sns.color_palette('Set2')

        fig1, ax1 = plt.subplots()
        ax1.pie(sex, labels=labels, autopct='%.0f%%',
                shadow=False, colors=color, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.subheader('Gender of New Joiners')
        st.pyplot(fig1)
##########################################################################################
#Starting Age
        st.subheader('Distribution of New Joiner Age')
        fig2, ax1 = plt.subplots()
        sns.histplot(data=df, x=df['Starting Age'],ax=ax1, hue='Gender', multiple='stack')
        st.pyplot(fig2)
##########################################################################################
#Starting Grade
        st.subheader('Distribution of New Joiner original grade in Taekwondo')
        fig3, ax1 = plt.subplots()
        sns.histplot(df['Starting Grade'],ax=ax1)
        st.pyplot(fig3)

############################################################################################
#Pie chart of Blackbelt as of Beginning of 2022
        B=df['BlackBelt'].value_counts().index
        st.subheader('% of BlackBelt Student as of Beginning of 2022')
        fig4, ax1 = plt.subplots()
        ax1.pie(x=df["BlackBelt"].value_counts(),labels=['Color Belt','Black Belt'],colors=color,autopct="%.1f%%")
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig4)
##########################################################################################

def BB():

        st.write("""
        # NRG New Joiner To Be Blackbelt Prediction App

        This app predicts the **Probability** to be a blackbelt!

        """)

        st.header('User Input Features')

        # Collects user input features into dataframe
        def user_input_features():
                gender = st.selectbox('Gender', ('male', 'female'))
                age = st.slider('Starting Age', 2, 65, step=1, value =5)  # numeric
                grade = st.slider('Starting Grade (white=10, black=1)', 1, 10, step=1, value=10)  # numeric
                if gender == 'male':
                        gender = 1
                else:
                        gender = 0
                data = {'Gender': gender,
                        'Age': age,
                        'Grade': grade}
                features = pd.DataFrame(data, index=[0])
                return features

        Xnew = user_input_features()

#######################################################################################

        Xnew=pd.DataFrame(Xnew)

        RF = pickle.load(open('RF.pickle', 'rb'))
        yRF=RF.predict_proba(Xnew)[:,1]


# Displays the user input features
        st.subheader('User Input features')

        if Xnew.iloc[0,0]==1:
                gender='M'
        else:
                gender='F'

        #Starting Age
        Age=Xnew.iloc[0,1]

        #Starting Grade
        G_Dict = {1: 'Black', 2: 'Red-Stripe', 3: 'Red', 4: 'Blue-Stripe', 5: 'Blue', 6: 'Green-Stripe', 7: 'Green',
                  8: 'Yellow-Stripe', 9: 'Yellow', 10: 'White'}

        G = G_Dict.get(Xnew.iloc[0, 2])

        st.write('Gender : ', gender)
        st.write('Age : ', str(Age))
        st.write('Starting Grade: ',G)

        st.subheader('Prediction Probability')

        st.write('Random Forest : %.0f'% ((yRF)*100),'%')

        if yRF >0.5:
                st.write("High Chance to achieve Blackbelt")
        else :
                st.write ("Fighting ~ You can do it!!!!")

####################################################################################################################


####################################################################################################################
if option == 'Statistics':
        stats()
elif option == 'Blackbelt Probability':
        BB()

elif option == 'Registration Forecast':
        st.write("""
        # Work in Progress """)
