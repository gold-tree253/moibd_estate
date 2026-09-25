import streamlit as st
import joblib
import pandas as pd

tab1, tab2, tab3 = st.tabs(["Расчёт цены", "Дашборд", "Справка"])

def _to_str(df):
    return df.astype(str)


import sklearn.compose._column_transformer

class _RemainderColsListStub:
    pass

sklearn.compose._column_transformer._RemainderColsList = _RemainderColsListStub


model = joblib.load('../data/best_model.pkl')

with tab1:
    st.title("Расчёт цены недвижимости")
    st.subheader("Введите характеристики")

    SUB_TYPES = [
        "Rezidans", "Daire", "Villa", "Müstakil Ev", "Kooperatif",
        "Yazlık", "Komple Bina", "Prefabrik Ev", "Köşk / Konak / Yalı",
        "Çiftlik Evi", "Yalı Dairesi", "Loft",
    ]
    HEATING_TYPES = [
        "Fancoil", "Yok", "Kalorifer (Doğalgaz)", "Kalorifer (Kömür)",
        "Kombi (Elektrikli)", "Klima", "Kombi (Doğalgaz)",
        "Merkezi Sistem (Isı Payı Ölçer)", "Merkezi Sistem",
        "Soba (Kömür)", "Yerden Isıtma", "Soba (Doğalgaz)",
        "Güneş Enerjisi", "Kalorifer (Akaryakıt)", "Jeotermal",
        "Kat Kaloriferi",
    ]

    sub_type = st.selectbox("Подтип недвижимости", SUB_TYPES)
    listing_type = st.text_input("Тип объявления (число 1 или 2)")
    tom = st.text_input("Время на рынке")
    size = st.text_input("Площадь м2")
    heating_type = st.selectbox("Тип отопления", HEATING_TYPES)
    building_age_num = st.text_input("Возраст здания (годы)")
    room_total = st.text_input("Всего комнат")
    floor_no_num = st.text_input("Этаж")
    total_floor_count_num = st.text_input("Всего этажей")
    city = st.text_input("Город")
    district = st.text_input("Район")

    if st.button("Рассчитать"):
        if sub_type == "":
            st.error("Ошибка: заполните подтип недвижимости")
        elif listing_type == "":
            st.error("Ошибка: заполните тип объявления")
        elif tom == "":
            st.error("Ошибка: заполните время на рынке")
        elif size == "" or city == "" or district == "":
            st.error("Ошибка: заполните площадь, город и район")
        else:
            try:
                listing_type_i = int(listing_type)
                tom_i = int(tom)
                size_f = float(size)
                building_age_num_f = float(building_age_num)
                room_total_f = float(room_total)
                floor_no_num_f = float(floor_no_num)
                total_floor_count_num_f = float(total_floor_count_num)

                if room_total_f < 0 or room_total_f > 20:
                    st.error("Ошибка: количество комнат должно быть от 0 до 20")
                elif floor_no_num_f < 0:
                    st.error("Ошибка: этаж не может быть отрицательным")
                elif total_floor_count_num_f <= 0:
                    st.error("Ошибка: количество этажей должно быть больше 0")
                elif floor_no_num_f > total_floor_count_num_f:
                    st.error("Ошибка: этаж не может быть больше общего числа этажей")
                elif building_age_num_f < 0:
                    st.error("Ошибка: возраст здания не может быть меньше 0")
                else:
                    data = {
                        "type": ["Konut"],
                        "sub_type": [sub_type],
                        "listing_type": [listing_type_i],
                        "tom": [tom_i],
                        "size": [size_f],
                        "heating_type": [heating_type],
                        "price_currency": ["TRY"],
                        "building_age_num": [building_age_num_f],
                        "room_total": [room_total_f],
                        "floor_no_num": [floor_no_num_f],
                        "total_floor_count_num": [total_floor_count_num_f],
                        "city": [city],
                        "district": [district],
                    }

                    X_new = pd.DataFrame(data)
                    for c in X_new.select_dtypes(include=["float64"]).columns:
                        X_new[c] = X_new[c].astype("float32")
                    pred = model.predict(X_new)
                    st.success("Цена: " + str(round(float(pred[0]), 2)) + " try")

            except Exception as e:
                st.error(e)




with tab2:
    st.markdown('### [Дашборд](https://datalens.yandex/xzkxu2l5bpl0g?_share_link=public)')




with tab3:
    st.title("Справка")
    st.write("Это приложение позволяет создать предварительный расчёт цены недвижимости на основе характеристик этой недвижимости. Валюта - турецкая лира. Чтобы получить цену, заполните все поля и нажмите кнопку 'Рассчитать'.")
    st.write("Вам нужно ввести следующие характеристики:")
    st.write("• Подтип недвижимости")
    st.write("• Тип объявления (число 1 или 2)")
    st.write("• Время на рынке")
    st.write("• Площадь м2")
    st.write("• Возраст здания (годы)")
    st.write("• Всего комнат")
    st.write("• Этаж")
    st.write("• Всего этажей")
    st.write("• Город")
    st.write("• Район")
    st.write("")

    st.divider()

    st.write("Используемая модель для расчёта цены - Bagging. Bagging - это ансамблевый метод, который объединяет прогнозы из нескольких методов обучения вместе, чтобы предсказывать более точно. Идея бэгинга заключается в том, что каждый базовый алгоритм обучается на случайном подмножестве обучающей выборки. В этом случае, даже используя одну модель алгоритмов, получаются различные базовые алгоритмы.")
    st.write("Плюсы:")
    st.write("• Снижает переобучение и улучшает точность")
    st.write("• Снижение дисперсии (variance)")
    st.write("• Хорошо работает с категориальными и непрерывными значениями")
    st.write("• Работает с высокоразмерными данными")
    st.write("Минусы:")
    st.write("• Не дает точных непрерывных прогнозов в задачах регрессии")
    st.write("• Управление моделью малое")
    st.write("• Требует много времени для обучения модели")

    st.divider()

    st.write("Авторы проекта: Тухватуллин Р.Р., Янгиров Т.Р.")
