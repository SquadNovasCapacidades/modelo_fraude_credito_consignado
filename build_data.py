import sys
import pandas as pd
from sklearn.externals import joblib


def import_base(data):
    try:
        data = pd.read_csv(data, encoding='latin')
    except:
        data = pd.read_excel(data, encoding='latin')

    colunas = ['DATA_SOLICITACAO', 'CONTRATO', 'CLIENTE', 'ORIGEM', 'CONVENIO', 'PRODUTO',
               'PRAZO', 'TAXA_MENSAL', 'PASTINHA', 'LOJA', 'CPF_CLIENTE', 'CIDADE_LOJA',
               'UF_LOJA', 'DIGITADOR', 'FORMA_CREDITO', 'SEXO', 'EST_CIVIL', 'DATA_NASCIMENTO',
               'IDADE', 'DATA_FALECIMENTO', 'CIDADE_CLI', 'UF_CLI', 'END_CEP', 'BANCO_CRED',
               'AGENCIA_CRED', 'DIG_AGENCIA_CRED', 'CONTA_CRED', 'DIG_CONTA_CRED', 'VALOR_LIQUIDO']
    dif_cols = [col for col in colunas if col not in data.columns]
    if len(dif_cols) > 0: raise ValueError('Colunas missing')
    size = len(data)
    data = data.dropna()
    print('Numero de amostras excluidas:', size - len(data))
    return data


def fix_columns(data):
    # Evitar erro na base
    data['CIDADE_CLI'] = data.CIDADE_CLI.apply(lambda x: x.split('      ')[0])
    data['CIDADE_LOJA'] = data.CIDADE_LOJA.apply(lambda x: x.split('      ')[0])
    data['ORIGEM'] = data.ORIGEM.apply(lambda x: x.split('   ')[0])
    data['FORMA_CREDITO'] = data.FORMA_CREDITO.apply(lambda x: x.split('   ')[0])

    return data


def get_freq(x, dici):
    try:
        return dici[x]
    except:
        return 1


def get_date(x):
    mes = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
           'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
    z = x[2:5]
    for k in mes:
        z = z.replace(k, mes[k])
    return x[5:] + '-' + z + '-' + x[0:2]


def create_variables(data):
    ### Cria variaveis cidades e estados diferentes para cliente e loja
    data['is_city'] = (data.CIDADE_LOJA == data.CIDADE_CLI).astype(int)
    data['is_UF'] = (data.UF_LOJA == data.UF_CLI).astype(int)

    ## Cria variaveis com datas

    # Get day of week
    data['day_week'] = pd.DatetimeIndex(data.DATA_SOLICITACAO).dayofweek
    # Get day
    data['day'] = pd.DatetimeIndex(data.DATA_SOLICITACAO).day
    # Get if is end of a month
    data['is_end_month'] = (data.day >= 15).astype(int)

    ## Cria variaveis com frequencia
    dici_lista = joblib.load('models/dici_lista.pkl')

    colunas = [[0, 'PRODUTO'], [1, 'LOJA'], [2, 'BANCO_CRED'], [3, 'CIDADE_LOJA'], [4, 'FORMA_CREDITO'],
               [5, 'CONVENIO'],
               [6, 'UF_LOJA'], [7, 'CIDADE_CLI'], [8, 'UF_CLI'], [9, 'END_CEP'], [10, 'PASTINHA'], [11, 'DIGITADOR']]
    for ind, col in colunas:
        coluna = col + '_freq'
        dici = dici_lista[ind]
        data[coluna] = data[col].apply(lambda x: get_freq(x, dici))

    # Dropa colunas
    data = data.drop(['PRODUTO', 'PASTINHA', 'DIGITADOR', 'DATA_NASCIMENTO', 'DATA_FALECIMENTO',
                      'CIDADE_LOJA', 'TAXA_MENSAL', 'CIDADE_CLI', 'DATA_SOLICITACAO', 'CPF_CLIENTE',
                      'AGENCIA_CRED', 'DIG_AGENCIA_CRED', 'CONTA_CRED', 'DIG_CONTA_CRED'], axis=1)
    return data


def fill_gap(data, colunas_treino):
    colunas = data.columns
    drop = [coluna for coluna in colunas if coluna not in colunas_treino]
    add = [coluna for coluna in colunas_treino if coluna not in colunas]
    for col in add:
        data[col] = 0
    data = data.drop(drop, axis=1)
    data = data[colunas_treino]
    return data


def get_dummies(data):
    data = pd.get_dummies(data, columns=['ORIGEM', 'UF_LOJA', 'FORMA_CREDITO', 'SEXO', 'EST_CIVIL', 'UF_CLI'])
    colunas_treino = joblib.load('models/train_columns.pkl')
    colunas_treino.append('CONTRATO')
    colunas_treino.append('CLIENTE')
    data = fill_gap(data, colunas_treino)
    return data


def main_prep(file):
    data = import_base(file)
    data = fix_columns(data)
    data = create_variables(data)
    data = get_dummies(data)
    data.to_csv('output/prepared_data.csv', index=False)
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception

    # source_file_path = 'https://s3.amazonaws.com/case-fcconsig/base_fcc.csv'
    source_file_path = sys.argv[1]
    main_prep(file=source_file_path)