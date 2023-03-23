
'''Dosya Python içine alınmadan önce ilgili kütüphaler import edilir.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 

'''Veri seti pandas kütüphanesinin pd.methodu ile okunur.Veri seti maaşların yıllara göre gösterildiği 2 ayrı kolondan oluşmaktadır.'''

veriler=pd.read_csv('Salary_Data.csv')
print(veriler)

'''Veri setinde maaşların yıllara göre gösterimi yer almış olup maaş bağımlı değişken iken yıllar ise bağımsız değişkendir.
Veri setinde yıl arttığında maaşlarda da artış gözlendiğinden dolayı aralarında doğrusal bir ilişki olduğundan bahsedilebilir.
Bağımlı ve bağımsız değişkenler belirlenir.'''

bagımsız_x=veriler.iloc[:, :1]
bagımlı_y=veriler.iloc[:, 1:]


'''Veri setinden elde edilen bağımlı ve bağımsız değişkenler test ve eğitim için bölünür.Verilerin %33'lük kısmı test edilmek için kullanılırken kalan kısım eğitiminde 
kullanılır.'''

from sklearn.model_selection import train_test_split
bagımsız_x_train,bagımsız_x_test,bagımlı_y_train,bagımlı_y_test=train_test_split(bagımsız_x,bagımlı_y,test_size=0.33,random_state=0)

'''Model eğitme aşamasına geçilir.Kullanılacak model Lineer regresyon modeli olup bu model makine öğrenmesinde kullanılan tahmin yöntemidir.Lineer regresyon,
bağımlı ve bağımsız değişkenler arasındaki ilişkiyi göstermek için kullanılan istatistiksel bir yöntemdir.Sürekli değişkenler arasındaki ilişkiyi analiz eden 
bir yöntem olan lineerr regresyonda belirli varsayımların bulunması halinde makine öğrenmesi için kullanımı uygun olmaktadır.
1)Değikenler arasında doğrusal bir ilişki olmalıdır.
2)Hata terimleri birbirinden bağımsız olmalıdır.
3)Hata terimleri,normal dağılım göstermelidir.Ancak bu şekilde model güvenilir olmaktadır.
4)Hata terimlerinin varyansı aynı olmalıdır.
5)Hata terimleri arasında otokorelasyon bulunmamalıdır.Bunun anlamı şudur:Hata terimlerinin birbirlerini etkileme imkanı olmamalıdır.
6)Eksik veriler olmamalıdır.'''

'''Bağımlı ve bağımsız değişkenler saçılım grafiği çizdirildiğinde aralarındaki doğrusal ilişki grafikte görülmektedir.'''

plt.scatter(bagımsız_x.values,bagımlı_y.values)
plt.xlabel('x')
plt.ylabel('Y')
plt.title('X y arasındaki ilişki')
plt.show()


'''Lineer regresyon modeli kuracağımız için sklearn kütüphanesinden Lineer Regresyon sınıfını import etmemiz gerekir.Sonrasında Lineer regresyon sınıfından bir nesne
oluşturup modeli fit methodu ile eğitilmelidir.'''

from sklearn.linear_model import LinearRegression
egitim=LinearRegression()
egitim.fit(bagımsız_x_train,bagımlı_y_train)

'''Model predict methodu ile tahminde bulunduktan sonra tahminleri tahmin objesine atanır.'''

tahmin=egitim.predict(bagımsız_x_test)

'''Regresyon modellerinin başarısını ölçmek için R^2 yöntemi kullanılır.Kullanılan modelin ne kadar başarılı olduğuna dair sayısal bir değer döndürür.
Bağımlı değişkenin ne kadarının bağımsız değişkenler tarafından açıklandığını gösterir.
R^2 değeri 0 ile 1 arasında bir değer almaktadır.Değer 1'e yaklaştıkça modelin başarı performansı da iyileşmektedir.'''

from sklearn.metrics import r2_score
r_2=r2_score(bagımlı_y_test, tahmin)
print('r_2 degeri')
print(r_2)

