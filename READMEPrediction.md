# Machine-Learning-Prediction-
Prediction Algorithm
## Lineer Regresyon
Bu projede, lineer regresyon modeli kullanarak maaş verileri üzerinde tahminler yapmayı amaçlıyoruz.

## Veri Seti
Veri setimiz "Salary_Data.csv" dosyasında yer almaktadır. Bu veri setinde maaşların yıllara göre gösterildiği 2 ayrı kolondan oluşmaktadır.

## Bağımlı ve Bağımsız Değişkenler
Veri setinde maaşların yıllara göre gösterimi yer almış olup maaş bağımlı değişken iken yıllar ise bağımsız değişkendir. Veri setinde yıl arttığında maaşlarda da artış gözlendiğinden dolayı aralarında doğrusal bir ilişki olduğundan bahsedilebilir.

## Veri Ön İşleme
Veri setinden elde edilen bağımlı ve bağımsız değişkenler test ve eğitim için bölünür. Verilerin %33'lük kısmı test edilmek için kullanılırken kalan kısım eğitiminde kullanılır.

## Model Eğitimi
Model eğitme aşamasında kullanılacak model Lineer regresyon modeli olup bu model makine öğrenmesinde kullanılan tahmin yöntemidir. Lineer regresyon, bağımlı ve bağımsız değişkenler arasındaki ilişkiyi göstermek için kullanılan istatistiksel bir yöntemdir.

Bağımlı ve bağımsız değişkenler saçılım grafiği çizdirildiğinde aralarındaki doğrusal ilişki grafikte görülmektedir.

Lineer regresyon modeli kuracağımız için sklearn kütüphanesinden LinearRegression sınıfını import etmemiz gerekir. Sonrasında LinearRegression sınıfından bir nesne oluşturup modeli fit methodu ile eğitilmelidir.

Model predict methodu ile tahminde bulunduktan sonra tahminleri tahmin objesine atanır.

## Model Performansı
Regresyon modellerinin başarısını ölçmek için R^2 yöntemi kullanılır. Kullanılan modelin ne kadar başarılı olduğuna dair sayısal bir değer döndürür. Bağımlı değişkenin ne kadarının bağımsız değişkenler tarafından açıklandığını gösterir. R^2 değeri 0 ile 1 arasında bir değer almaktadır. Değer 1'e yaklaştıkça modelin başarı performansı da iyileşmektedir.

## Kullanılan Kütüphaneler
pandas
numpy
matplotlib
sklearn

## Örnek Kullanım

```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

# Veri seti okuma
veriler=pd.read_csv('Salary_Data.csv')




