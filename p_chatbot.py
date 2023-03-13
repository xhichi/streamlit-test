import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import psutil

@st.cache_data
def cached_model():
    model=SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache_data
def get_dataset():
    df=pd.read_csv('wellness_dataset.csv')
    df['embedding']=df['embedding'].apply(json.loads)
    return df

def get_hw_idle_info():
    rst = dict()
    
    cp = psutil.cpu_times_percent(interval=None, percpu=False)
    cp_item = dict()
    cp_item['free'] = psutil.cpu_count(logical=False) * (cp.idle/100)
    cp_item['idle'] = cp.idle
    cp_item['desc'] = f"Idle CPU: {cp_item['free']:.2f} core ({cp_item['idle']}%)"
    print(cp_item['desc'])

    vm = psutil.virtual_memory()
    vm_item = dict()
    vm_item['free'] = vm.available//(1024*1024)
    vm_item['idle'] = vm.available/vm.total*100
    vm_item['desc'] = f"Idle Memory: {vm_item['free']:,}MB ({vm_item['idle']:.1f}%)"
    print(vm_item['desc'])
    
    du = psutil.disk_usage(path='/')
    du_item = dict()
    du_item['free'] = du.free//(1024*1024)
    du_item['idle'] = du.free/du.total*100
    du_item['desc'] = f"Idle Disk: {du_item['free']:,}MB ({du_item['idle']:.1f}%)"
    print(du_item['desc'])

model=cached_model()
df=get_dataset()

st.header('심리상담')
st.markdown('파이 챗봇')

if 'generated' not in st.session_state:
    st.session_state['generated']=[]

if 'past' not in st.session_state:
    st.session_state['past']=[]

with st.form('form', clear_on_submit=True):
    user_input=st.text_input('고객님:', '')
    submitted=st.form_submit_button('전송')

if submitted and user_input:
    embedding=model.encode(user_input)
    df['similarity']=df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer=df.loc[df['similarity'].idxmax()]
    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])
    get_hw_idle_info()

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
    if len(st.session_state['generated'])>i:
        message(st.session_state['generated'][i], key=str(i)+'_bot')
