# -*- coding: utf-8 -*-
import json, os, joblib
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
st.set_page_config(page_title='Прогноз риска ССЗ', page_icon='❤️', layout='wide')
STUDENT_NAME='Рустемов Шахмурат'; ADVISOR_NAME='Садуов Алишер Берикжанович'
with st.sidebar:
    st.markdown('### О проекте')
    st.markdown(f'Этот проект выполнен учеником **{STUDENT_NAME}** под руководством **{ADVISOR_NAME}**.')
    st.markdown('Модель: интерпретируемые ML-подходы (LR / DT / RF).')
    st.caption('Сервис ознакомительный, не заменяет консультацию врача.')

@st.cache_resource
def load_artifacts():
    model_path='artifacts/best_model.pkl'; meta_path='artifacts/metadata.json'
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error('Не найдены artifacts/best_model.pkl или artifacts/metadata.json'); st.stop()
    model=joblib.load(model_path)
    with open(meta_path,'r',encoding='utf-8') as f: meta=json.load(f)
    return model, meta
model, meta = load_artifacts()
FEATURE_ORDER = meta['feature_order']

HELP={'age':('лет',18,100,1),'trestbps':('мм рт. ст. (АД в покое)',80,200,1),'chol':('мг/дл (общий холестерин)',100,600,1),
       'thalach':('уд/мин (макс. ЧСС на нагрузке)',60,220,1),'oldpeak':('ед. (депрессия ST)',0.0,6.5,0.1),'ca':('сосуды',0,3,1)}
CP={0:'0 — типичная стенокардия',1:'1 — атипичная стенокардия',2:'2 — неангинальная боль',3:'3 — бессимптомно'}
ECG={0:'0 — нормальная ЭКГ',1:'1 — ST-T аномалии',2:'2 — гипертрофия ЛЖ'}
SL={0:'0 — восходящий',1:'1 — плоский',2:'2 — нисходящий'}
TH={0:'0 — нормальная',1:'1 — фикс. дефект',2:'2 — обратимый дефект',3:'3 — не описано'}

st.markdown('## Прогнозирование риска сердечно-сосудистых заболеваний')
st.markdown('Введите параметры пациента и получите вероятность риска и рекомендации.')

with st.form('form'):
    c1,c2,c3=st.columns(3)
    with c1:
        age=st.number_input(f'Возраст, {HELP["age"][0]}',min_value=HELP['age'][1],max_value=HELP['age'][2],value=52,step=HELP['age'][3])
        trestbps=st.number_input(f'АД (trestbps), {HELP["trestbps"][0]}',min_value=HELP['trestbps'][1],max_value=HELP['trestbps'][2],value=130,step=HELP['trestbps'][3])
        chol=st.number_input(f'Холестерин (chol), {HELP["chol"][0]}',min_value=HELP['chol'][1],max_value=HELP['chol'][2],value=240,step=HELP['chol'][3])
    with c2:
        thalach=st.number_input(f'Макс. ЧСС, {HELP["thalach"][0]}',min_value=HELP['thalach'][1],max_value=HELP['thalach'][2],value=150,step=HELP['thalach'][3])
        oldpeak=st.number_input(f'Oldpeak, {HELP["oldpeak"][0]}',min_value=HELP['oldpeak'][1],max_value=HELP['oldpeak'][2],value=1.0,step=HELP['oldpeak'][3])
        ca=st.number_input(f'Сосуды (ca), {HELP["ca"][0]}',min_value=HELP['ca'][1],max_value=HELP['ca'][2],value=0,step=1)
    with c3:
        sex=st.selectbox('Пол (sex)',[0,1],format_func=lambda x:'0 — женщина' if x==0 else '1 — мужчина')
        fbs=st.selectbox('Fasting blood sugar > 120 (fbs)',[0,1],format_func=lambda x:'0 — нет' if x==0 else '1 — да')
        exang=st.selectbox('Стресс-стенокардия (exang)',[0,1],format_func=lambda x:'0 — нет' if x==0 else '1 — да')
    c4,c5,c6,c7=st.columns(4)
    with c4: cp=st.selectbox('Боль в груди (cp)',list(CP.keys()),format_func=lambda k:CP[k])
    with c5: restecg=st.selectbox('ЭКГ (restecg)',list(ECG.keys()),format_func=lambda k:ECG[k])
    with c6: slope=st.selectbox('Наклон ST (slope)',list(SL.keys()),format_func=lambda k:SL[k])
    with c7: thal=st.selectbox('Thal',list(TH.keys()),format_func=lambda k:TH[k])
    st.divider()
    th1,th2=st.columns([1,2])
    with th1: threshold=st.slider('Порог классификации',0.1,0.9,0.5,0.05)
    with th2: st.caption('Для скрининга порог можно снизить ради повышения чувствительности (Recall).')
    submit=st.form_submit_button('Рассчитать риск',type='primary')

def make_df():
    data={'age':age,'trestbps':trestbps,'chol':chol,'thalach':thalach,'oldpeak':oldpeak,'ca':ca,
           'sex':int(sex),'fbs':int(fbs),'exang':int(exang),'cp':int(cp),'restecg':int(restecg),'slope':int(slope),'thal':int(thal)}
    df=pd.DataFrame([data]); return df[FEATURE_ORDER]

def gauge(prob):
    fig=go.Figure(go.Indicator(mode='gauge+number',value=prob*100,
        number={'suffix':'%'},title={'text':'Вероятность ССЗ'},
        gauge={'axis':{'range':[0,100]},'bar':{'color':'crimson'},
                'steps':[{'range':[0,30],'color':'#c9f7c5'},{'range':[30,60],'color':'#ffe082'},{'range':[60,100],'color':'#ffcccb'}]})
    )
    fig.update_layout(height=260,margin=dict(l=10,r=10,t=40,b=10)); return fig

def recs(x,prob):
    R=[]
    R.append('Высокий риск: обратитесь к кардиологу в ближайшее время.' if prob>=0.60
             else ('Умеренный риск: обсудите дообследование.' if prob>=0.30
                   else 'Низкий риск: контроль показателей и ЗОЖ.'))
    if x['trestbps']>=140: R.append('Повышено АД — нужен контроль и терапия по показаниям.')
    if x['chol']>=240: R.append('Высокий холестерин — диета/нагрузка, оценка терапии.')
    if int(x['fbs'])==1: R.append('Гликемия натощак >120 — консультация эндокринолога.')
    if int(x['exang'])==1: R.append('Стенокардия при нагрузке — очная оценка.')
    if x['oldpeak']>=1.0: R.append('Депрессия ST — возможна ишемия.')
    if x['age']>=55: R.append('Возрастной фактор — регулярные чекапы.')
    R.append('Дисклеймер: модель не ставит диагноз, решение принимает врач.')
    return R

if submit:
    X=make_df(); prob=float(model.predict_proba(X)[:,1][0]); yhat=int(prob>=threshold)
    L,R=st.columns([1.2,1])
    with L:
        st.subheader('Результат')
        st.write(f'**Вероятность ССЗ:** {prob*100:.1f}%  \n**Порог:** {threshold:.2f} → **Класс:** {"Положительный" if yhat==1 else "Отрицательный"}')
        st.plotly_chart(gauge(prob), use_container_width=True)
    with R:
        st.subheader('Краткая справка')
        st.write(f'- Возраст: **{age}**  \n- Пол: **{"мужчина" if int(sex)==1 else "женщина"}**  \n- АД: **{trestbps}** мм рт. ст.  \n- Холестерин: **{chol}** мг/дл  \n- Макс. ЧСС: **{thalach}** уд/мин  \n- Oldpeak: **{oldpeak}**  \n- Сосуды (ca): **{ca}**')
    st.subheader('Рекомендации')
    for r in recs(X.iloc[0], prob): st.markdown(f'- {r}')
