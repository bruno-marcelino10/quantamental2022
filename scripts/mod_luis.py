n = 100
datas = list(df.index)
n_datas = len(datas)

def treino(df):
	# df contem variaveis, alvo e datas
	
	trains, tests = [], []
	for i in range(len(datas) - n): # 1700
		
		train = df.iloc[i:i+n, :]				
		trains.append(train)
		test = df.iloc[i+n, :]
		tests.append(test)
		
	return trains, tests
		
def previsao(train, test):
	
	data_obs = test.index
	X_train = train.drop("Alvo")
	y_train = train["Alvo"]
	X_test = test.drop("Alvo")
	y_test = test["Alvo"]
	
	modelo = MLPClassifier(random_state = 42, solver = "sgd", activation = "tanh")
	modelo.fit(X_train, y_train)
	y_pred = modelo.predict(X_test, y_test)
	
	bt = pd.DataFrame()
	bt["y_t"] = y_test
	bt["y_p"] = pd.Series(y_pred)
	
	
	ret = np.where(bt["y_p"] == 1, bt["Alvo_Continuo"], -bt["Alvo_Continuo"])
	print(ret)
	
	return ret
	
retornos = []
for i, j in trains, tests:
	retornos.append(previsao(i,j))
	
(retornos.cumprod()).plot()