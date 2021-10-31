import numpy as np
import pandas as pd

df = pd.read_csv('D:/study/3-2/머러/LAB2/housing.csv')

print(df.head())
print(df.info())
print(df.describe().transpose())
print(df.isna().sum())

# Handling the Missing Values
import math
total_bedrooms_median = math.floor(df["total_bedrooms"].median())
print(total_bedrooms_median)

df["total_bedrooms"] = df["total_bedrooms"].fillna(total_bedrooms_median)
print(df.isna().sum())

print(df['ocean_proximity'])

# Convert the categorical data of "ocean_proximity" column
from sklearn.preprocessing import LabelEncoder
l_er = LabelEncoder()
df['ocean_proximity'] = l_er.fit_transform(df['ocean_proximity'])

print(df['ocean_proximity'])


from matplotlib import pyplot as plt
import seaborn as sns

colormap = plt.cm.PuBu
plt.figure(figsize=(10, 8))
plt.title("Person Correlation of Features", y = 1.05, size = 15)
sns.heatmap(df.astype(float).corr(), linewidths = 0.1, vmax = 1.0, square = True,
            cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 16})
plt.show()


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def K_means_plot(dataset, cluster_lists, scaler):

    from sklearn.cluster import KMeans

    pca = PCA(2)
    dataset = pca.fit_transform(dataset)

    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)
    fig.subplots_adjust(top=0.8)


    for ind, n_cluster in enumerate(cluster_lists):
        model = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        km_labels = model.fit_predict(dataset)
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n')

        # plot the input data
        u_labels = np.unique(km_labels)
        centroids = model.cluster_centers_
        for i in u_labels:
            axs[ind].scatter(dataset[km_labels == i, 0], dataset[km_labels == i, 1], label=i)

        # Labeling the clusters
        # Draw white circles at cluster centers
        axs[ind].scatter(centroids[:, 0], centroids[:, 1], marker='o',
                    c="white", alpha=1, s=100, edgecolor='k')
        for i, c in enumerate(centroids):
            axs[ind].scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=25, edgecolor='k')
    plt.suptitle(scaler, fontsize=12, fontweight='bold', y=0.98)
    plt.show()

# Get an accuracy for each models for each dataset using each scaling method.
sse = []
silhouette_avg_n_clusters = []

## 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def kmeans_silhouette_eblow(cluster_lists, X_features, scaler):

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    sse.clear()
    silhouette_avg_n_clusters.clear()

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    fig.subplots_adjust(top=0.8)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):

        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산.
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        sse.append(clusterer.inertia_)
        print('[Running] : {:.2f}%'.format(n_cluster/len(cluster_lists)*100))

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        silhouette_avg_n_clusters.append(sil_avg)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
    plt.suptitle(scaler, fontsize=12, fontweight='bold', y=0.98)
    plt.show()
    # 실루엣 평균값이 제일큰 것을 plot
    # K_means_plot(X_features, silhouette_avg_n_clusters.index(max(silhouette_avg_n_clusters))+2)
    elbowPlot(X_features, scaler)
    K_means_plot(X_features, range(2, 13), scaler)

## 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def gmm_silhouette(cluster_lists, X_features, scaler):

    from sklearn.mixture import GaussianMixture as GMM
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    sse.clear()
    silhouette_avg_n_clusters.clear()

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    fig.subplots_adjust(top=0.8)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):

        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산.
        clusterer = GMM(n_components = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)

        print('[Running] : {:.2f}%'.format(n_cluster/len(cluster_lists)*100))

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        silhouette_avg_n_clusters.append(sil_avg)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
    plt.suptitle(scaler, fontsize=12, fontweight='bold', y=0.98)
    plt.show()


def elbowPlot(dataset, scaler):
    # pip install --upgrade kneed
    from kneed import KneeLocator
    kl = KneeLocator(
        range(2, 13), sse, curve="convex", direction="decreasing"
    )
    kl.plot_knee()
    plt.title(scaler)
    plt.show()

    print('elbow:', kl.elbow)
    # K_means_plot(dataset, kl.elbow, scaler)

def gmm_cluster(dataset, scaler):
    from sklearn.mixture import GaussianMixture as GMM
    import matplotlib.pyplot as plt

    pca = PCA(2,whiten=True)
    dataset = pca.fit_transform(dataset)
    n_components = np.arange(50, 210, 10)
    models = [GMM(n, covariance_type='full', random_state=0) for n in n_components]
    aics = [model.fit(dataset).aic(dataset) for model in models]

    plt.plot(n_components, aics)
    plt.ylabel('AIC')
    plt.xlabel('Number of Components')
    plt.title('AIC for the number of GMM components')
    plt.axhline(y=min(aics), color="red", linestyle="--")
    plt.show()

    gmm = GMM(n_components[aics.index(min(aics))],covariance_type='full', random_state=0)
    gmm.fit(dataset)
    print('{} / {} Converged : {}'.format(scaler, n_components[aics.index(min(aics))], gmm.converged_))

    gmm_silhouette(range(2,13), dataset, scaler)

    plot_gmm(gmm, dataset)
    plt.title('{} \ GMM Clustering with n_components :{}'.format(scaler, gmm.n_components))
    plt.show()


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap=plt.cm.get_cmap('rainbow', 200), zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')


def clustering_clarans(select_df): #clarans clustering
    from pyclustering.cluster.clarans import clarans
    from pyclustering.cluster.silhouette import silhouette
    from pyclustering.cluster import cluster_visualizer
    from pyclustering.cluster.kmeans import kmeans
    import matplotlib.pyplot as plt

    sil_clarans = []
    visualizer = cluster_visualizer(10, 4) #시각적 그래프

    wce=[]
    def elbow_cal_wce(): #elbow wce 계산 -> pyclustering.cluster.elbow code 참조
        centers=select_df[df_medoids] #clarans의 center의 좌표주소
        instance = kmeans(select_df, centers, ccore=False) #center 좌표 주소에 대한 clustering
        instance.process()
        wce.append(instance.get_total_wce()) #wce 결과 저장

    def cal_elbows(): #계산된 wce로 distance 계산 -> pyclustering.cluster.elbow code 참조
        __elbows = []
        x0, y0 = 0.0, wce[0]
        x1, y1 = float(len(wce)), wce[-1]

        for index_elbow in range(1, len(wce) - 1):
            x, y = float(index_elbow), wce[index_elbow]

            segment = abs((y0 - y1) * x + (x1 - x0) * y + (x0 * y1 - x1 * y0))
            norm = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            distance = segment / norm

            __elbows.append(distance) #distance 저장
        return __elbows

    for k in range(3, 13): #k값 3~12 -> gridsearchcv 사용불가
        df_clarans = clarans(select_df, k, 1, 1) #num_local, maxneighbor 값 1로 설정 -> 시간이 오래걸림.
        df_clarans.process()
        df_medoids = df_clarans.get_medoids() #center 가져오기 -> center 좌표값 출력하려면 select_df[df_medoid] 사용할 것
        df_cluster = df_clarans.get_clusters() #cluster 가져오기
        score = silhouette(select_df, df_cluster).process().get_score() #실루엣 score 계산 -> 각 포인트마다의 score가 계산됨
        sil_clarans.append(np.nanmean(score)) #nan값 제외하고 평균값 저장

        elbow_cal_wce() #elbow 계산

        if len(select_df[0])!=4: #4차원 시각적 표현 불가
            visualizer.append_clusters(df_cluster, select_df,k-3) #cluster 시각화
            visualizer.append_cluster(df_medoids, select_df,k-3, marker='x') #center 좌표 시각화
            visualizer.set_canvas_title(text="Clarans Cluster : " + str(k), canvas=k-3)

    if len(select_df[0]) != 4:  # 4차원 시각적 표현 불가
        visualizer.show(figure=plt.figure(figsize=(8,6)))

    print("Clarans Silhouette Best score : "+str(np.max(sil_clarans))) #실루엣 score
    print("Clarans Silhouette Best cluster : " + str(np.argmax(sil_clarans)+3)) #실루엣 cluster
    plt.title('Clarans Silhouette')
    plt.plot(range(3, 13), sil_clarans)
    plt.show()

    print("Clarans Best cluster's Elbow WCE : " + str(wce[np.argmax(sil_clarans)]))  # Best cluster's elbow wce
    plt.title('Clarans Elbow WCE')
    plt.plot(range(3, 13), wce)
    plt.show()

    _elbow = cal_elbows()
    plt.title('Clarans Elbow Distance')
    plt.plot(range(3, 11), _elbow)
    plt.show()
    wce = []

silhouette_avg = []

def meanshiftProcess(dataset):
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np

    silhouette_avg.clear()
    dataset = dataset[:5000]
    samples = [50, 100, 150, 200, 250]
    fig, axs = plt.subplots(figsize=(4*5, 4), nrows=1, ncols=5)
    fig.subplots_adjust(top=0.8)
    for ind, k in enumerate(samples):
        print(k)
        bandwidth = estimate_bandwidth(dataset, quantile=0.2, n_samples=k, n_jobs=-1)
        ms = MeanShift(bandwidth=bandwidth)
        cluster = ms.fit(dataset)
        ms_labels = ms.fit_predict(dataset)
        u_labels = np.unique(cluster.labels_)

        sil_avg = silhouette_score(dataset, ms_labels)
        # centroids = ms.cluster_centers_
        for l in u_labels:
            axs[ind].scatter(dataset[ms_labels == l, 0], dataset[ms_labels == l, 1], label=l)

        axs[ind].legend([], [], frameon=False)
        axs[ind].set_title('Mean Shift with {} samples \n'
                  'Silhouette score : {:.4f}'.format(k, sil_avg ))
        silhouette_avg.append(sil_avg)
    plt.suptitle('{}/{}'.format(i, j) , fontsize=14, fontweight='bold', y=0.98)
    plt.show()
    plt.plot(samples, silhouette_avg)
    plt.title('Silhouette Score for the number of each samples')
    plt.show()


def findOptimalNClustersDB(dataset):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np

    # Number of clusters to search for and silhouette_scores list
    range_eps = [0.01, 0.05, 0.1]
    silhouette_avg.clear()

    fig, axs = plt.subplots(figsize=(4 * 5, 4), nrows=1, ncols=len(range_eps))
    fig.subplots_adjust(top=0.8)

    # Testing n_clusters options
    for ind, eps in enumerate(range_eps):
        # print(j, n_clusters, k, eps)
        db = DBSCAN(eps=eps, n_jobs=-1)
        cluster = db.fit(dataset)

        db_labels = db.fit_predict(dataset)
        u_labels = np.unique(cluster.labels_)
        print(db_labels)
        sil_avg = silhouette_score(dataset, db_labels)
        centroids = ms.cluster_centers_
        for l in u_labels:
            axs[ind].scatter(dataset[db_labels == l, 0], dataset[db_labels == l, 1], label=l)

        axs[ind].set_title('DBSCAN with eps {}\n'
                            'Silhouette score : {:.4f}'.format(eps, sil_avg))
        axs[ind].legend([], [], frameon=False)

        silhouette_avg.append(sil_avg)
        print('Silhouette Coefficient: {:.4f}'.format(sil_avg))
    plt.suptitle('{}/{}'.format(i, j), fontsize=14, fontweight='bold', y=0.98)
    plt.show()

    plt.plot(range_eps, silhouette_avg)
    plt.title('Silhouette Score for the number of each eps')
    plt.xlabel('eps')
    plt.ylabel('silhouette score')
    plt.show()

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler


predictor=[
        ['total_rooms', 'total_bedrooms'],
           ['population', 'households'],
           ['total_rooms', 'total_bedrooms', 'households'],
           ['total_rooms', 'total_bedrooms', 'households','population']]

# Declare what i will use scalers, models with various parameters.
scalers = [StandardScaler(), MaxAbsScaler(), MinMaxScaler(), RobustScaler()]

for i in predictor:
    print('----------------{}------------'.format(i))
    select_df = df[i]
    for j in scalers:
        print('----------------{}------------'.format(j))
        select_df = j.fit_transform(select_df)
        kmeans_silhouette_eblow(range(2, 13), select_df, j)
        gmm_cluster(select_df, j)
        clustering_clarans(select_df)
        findOptimalNClustersDB(select_df)
        meanshiftProcess(select_df)
