import streamlit as st
import openai
from streamlit_chat import message
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import streamlit.components.v1 as stc


openai.api_key = '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'


def main():
    st.title("MindPal Mental Health ChatBot")

    menu = ["Chat With Me!", "Take a mental health disorder questionairre!", "Emergency"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Chat With Me!":
        st.subheader("Hi! I am MindPal, your brain health buddy!")
        def generate_response(prompt):
            completions = openai.Completion.create (
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" Human:", " AI:"]
        )

            message = completions.choices[0].text
            return message
        
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []


        def get_text():
            #input_text = st.text_input("Human [enter your message here]: "," Hello Mr AI how was your day today? ", key="input")
            input_text= st.text_input('Human [enter your message here]:', '')
            return input_text 


        user_input = get_text()



        if user_input:
            output = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
            


        if st.session_state['generated']:

            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')




        # @st.cache_data(func={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash})
        # def load_data():    
        #     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        #     model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        #     return tokenizer, model
        # tokenizer, model = load_data()
        # st.write("Welcome to the Chatbot. I am still learning, please be patient")
        # input = st.text_input('User:')
        # if 'count' not in st.session_state or st.session_state.count == 6:
        #     st.session_state.count = 0 
        #     st.session_state.chat_history_ids = None
        #     st.session_state.old_response = ''
        # else:
        #     st.session_state.count += 1

        # new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')
        # bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids
        # st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

        # response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # if st.session_state.old_response == response:

        #     bot_input_ids = new_user_input_ids
        
        #     st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
        #     response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # st.write(f"Chatbot: {response}")

        # st.session_state.old_response = response
                 



        # message_history= []

        # message("My message") 

        # for message_ in message_history:
        #     message(message_)   # display all the previous message

        # placeholder = st.empty()  # placeholder for latest message
        # input_ = st.text_input("you:")
        # message_history.append(input_)

        # with placeholder.container():
        #     message(message_history[-1]) # display the latest message

    if choice == "Take a mental health disorder questionairre!":
        st.subheader("Mental Health Disorder Questionairre")

        HTML_BANNER = """
            <div style="background: 
            radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%);
            padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center;">Mental Health Questionairre</h1>
            </div>
            """
        stc.html(HTML_BANNER)

        def get_value(val, my_dict):
            for key, value in my_dict.items():
                if val == key:
                    return value
        
        st.subheader("Select the options to predict")


        #Depression
        st.subheader("In the past seven days...")

        felt_worthless = {"No": 0, "Yes": 1}
        choice_feltworthless = st.radio("Have you felt worthless?", tuple(felt_worthless.keys()))
        result_feltworthless = get_value(choice_feltworthless, felt_worthless)

        look_forward = {"No": 0, "Yes": 1}
        choice_lookforward = st.radio("Have you felt that you had nothing to look forward to?", tuple(look_forward.keys()))
        result_lookforward = get_value(choice_lookforward, look_forward)

        helpless = {"No": 0, "Yes": 1}
        choice_helpless = st.radio("Have you felt helpless?", tuple(helpless.keys()))
        result_helpless = get_value(choice_helpless, helpless)

        sad = {"No": 0, "Yes": 1}
        choice_sad = st.radio("Have you felt sad?", tuple(sad.keys()))
        result_sad = get_value(choice_sad, sad)

        failure = {"No": 0, "Yes": 1}
        choice_failure = st.radio("Have you felt like a failure?", tuple(failure.keys()))
        result_failure = get_value(choice_failure, failure)

        depressed = {"No": 0, "Yes": 1}
        choice_depressed = st.radio("Have you felt depressed?", tuple(depressed.keys()))
        result_depressed = get_value(choice_depressed, depressed)

        unhappy = {"No": 0, "Yes": 1}
        choice_unhappy = st.radio("Have you felt unhappy?", tuple(unhappy.keys()))
        result_depressed = get_value(choice_unhappy, unhappy)

        hopeless = {"No": 0, "Yes": 1}
        choice_hopeless = st.radio("Have you felt hopeless?", tuple(hopeless.keys()))
        result_hopeless = get_value(choice_hopeless, hopeless)

        #Anger
        st.subheader("In the past seven days...")

        irritated = {"No": 0, "Yes": 1}
        choice_irritated = st.radio("You were irritate more than people knew?", tuple(irritated.keys()))
        result_irritated = get_value(choice_irritated, irritated)

        angry = {"No": 0, "Yes": 1}
        choice_angry = st.radio("Have you felt angry?", tuple(angry.keys()))
        result_angry = get_value(choice_angry, angry)

        explode = {"No": 0, "Yes": 1}
        choice_explode = st.radio("Have you felt that your mind was ready to explode?", tuple(explode.keys()))
        result_explode = get_value(choice_explode, explode)
        
        grouchy = {"No": 0, "Yes": 1}
        choice_grouchy = st.radio("Have you felt grouchy?", tuple(grouchy.keys()))
        result_grouchy = get_value(choice_grouchy, grouchy)

        annoyed = {"No": 0, "Yes": 1}
        choice_annoyed = st.radio("Have you felt annoyed?", tuple(annoyed.keys()))
        result_annoyed = get_value(choice_annoyed, annoyed)


        #Anxiety
        st.subheader("In the past seven days...")

        fearful = {"No": 0, "Yes": 1}
        choice_fearful = st.radio("Have you felt fearful?", tuple(fearful.keys()))
        result_fearful = get_value(choice_fearful, fearful)
        
        anxious = {"No": 0, "Yes": 1}
        choice_anxious = st.radio("Have you felt anxious?", tuple(anxious.keys()))
        result_anxious = get_value(choice_anxious, anxious)

        worried = {"No": 0, "Yes": 1}
        choice_worried = st.radio("Have you felt worried?", tuple(worried.keys()))
        result_worried = get_value(choice_worried, worried)

        focus = {"No": 0, "Yes": 1}
        choice_focus = st.radio("Have you felt like you couldn't focus on anything else other than your anxiety?", tuple(focus.keys()))
        result_focus = get_value(choice_focus, focus)

        nervous = {"No": 0, "Yes": 1}
        choice_nervous= st.radio("Have you felt nervous?", tuple(nervous.keys()))
        result_nervous = get_value(choice_nervous, nervous)

        uneasy = {"No": 0, "Yes": 1}
        choice_uneasy = st.radio("Have you felt uneasy?", tuple(uneasy.keys()))
        result_uneasy = get_value(choice_uneasy, uneasy)

        tense = {"No": 0, "Yes": 1}
        choice_tense= st.radio("Have you felt tensed?", tuple(tense.keys()))
        result_tense = get_value(choice_tense, tense)

        #Sleep Disorder
        st.subheader("In the past seven days...")

        restless = {"No": 0, "Yes": 1}
        choice_restless = st.radio("Have you felt restless while sleeping?", tuple(restless.keys()))
        result_restless = get_value(choice_restless, restless)

        refreshing = {"No": 0, "Yes": 1}
        choice_refreshing = st.radio("Have you felt refreshed after your sleep?", tuple(refreshing.keys()))
        result_refreshing = get_value(choice_refreshing, refreshing)

        satisfied = {"No": 0, "Yes": 1}
        choice_satisfied = st.radio("Have you felt satisfied after your sleep?", tuple(satisfied.keys()))
        result_satisfied = get_value(choice_satisfied, satisfied)

        falling = {"No": 0, "Yes": 1}
        choice_falling = st.radio("Have you felt difficulty falling asleep?", tuple(falling.keys()))
        result_falling = get_value(choice_falling, falling)

        enough = {"No": 0, "Yes": 1}
        choice_enough= st.radio("Do you get enough sleep?", tuple(enough.keys()))
        result_enough = get_value(choice_enough, enough)

        #Substance dependence
        st.subheader("In the past seven days...")

        stimulants = {"No": 0, "Yes": 1}
        choice_stimulants= st.radio("Have you used stimulants like Ritalin or Adderall?", tuple(stimulants.keys()))
        result_stimulants = get_value(choice_stimulants, stimulants)

        painkillers = {"No": 0, "Yes": 1}
        choice_painkillers =  st.radio("Have you used painkillers like Vicodin?", tuple(painkillers.keys()))
        result_painkillers = get_value(choice_painkillers, painkillers)

        sedatives = {"No": 0, "Yes": 1}
        choice_sedatives =  st.radio("Have you used sedatives or tranquilizers like sleeping pills or Valium?", tuple(sedatives.keys()))
        result_sedatives = get_value(choice_sedatives, sedatives)

        marijuana = {"No": 0, "Yes": 1}
        choice_marijuana =  st.radio("Have you used marijuana?", tuple(marijuana.keys()))
        result_marijuana = get_value(choice_marijuana, marijuana)

        coke = {"No": 0, "Yes": 1}
        choice_coke =  st.radio("Have you used coke/cocaine or crack?", tuple(coke.keys()))
        result_coke = get_value(choice_coke, coke)

        ecstasy = {"No": 0, "Yes": 1}
        choice_ecstasy =  st.radio("Have you used club drugs like ecstasy?", tuple(ecstasy.keys()))
        result_ecstasy = get_value(choice_ecstasy, ecstasy)

        hallucinogens = {"No": 0, "Yes": 1}
        choice_hallucinogens =  st.radio("Have you used hallucinogens like LSD or DMT?", tuple(hallucinogens.keys()))
        result_hallucinogens = get_value(choice_hallucinogens, hallucinogens)

        heroin = {"No": 0, "Yes": 1}
        choice_heroin =  st.radio("Have you used heroin?", tuple(heroin.keys()))
        result_heroin = get_value(choice_heroin, heroin)

        inhalents = {"No": 0, "Yes": 1}
        choice_inhalents =  st.radio("Have you used inhalents or solvents like glue/solution?", tuple(inhalents.keys()))
        result_inhalents = get_value(choice_inhalents, inhalents)

        meth = {"No": 0, "Yes": 1}
        choice_meth=  st.radio("Have you used Methamphetamine like Speed?", tuple(meth.keys()))
        result_meth = get_value(choice_meth, meth)


        st.subheader("Prediction Tool")
        if st.checkbox("Make Prediction"):
            st.subheader("The prediction is: ")





if __name__ == '__main__':
    main()

        # from sklearn.neural_network import MLPClassifier
        # import joblib

        # st.subheader("Prediction Tool")
        # if st.checkbox("Make Prediction"):
        #     all_ml_dict = {
        #         'Neural Network-Multi-layer Perceptron Classifier': MLPClassifier()
        #     }

        #     # Find the Key From Dictionary
        #     def get_key(val, my_dict):
        #         for key, value in my_dict.items():
        #             if val == value:
        #                 return key
                    
        
        #  # Load Models
        #     def load_model_n_predict(model_file):
        #         loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
        #         return loaded_model

        # model_predictornnmlp = load_model_n_predict("mlp_model_pickle.joblib")
        # prediction = model_predictornnmlp.predict(disorder_data)

        # st.markdown('**Your disease prediction according to the inputs**')
        # st.subheader(prediction)







    

