基于Transformer实现人格个性指示

本项目基于Transformer实现Myers Briggs类型指示器，利用Transformer的编码器进行语义提取，通过训练对社交媒体帖子进行个性分类。数据集包含超过8600行MBTI类型数据，模型训练后能根据用户的社交内容预测其个性类型。



步骤
cd MBTI_Transformer

conda create -n mbti_env python=3.10

conda activate mbti_env  
虚拟环境 中安装 Hugging Face 相关的库
pip install transformers datasets torch scikit-learn pandas matplotlib



用 Jupyter Notebook 进行实验
pip install jupyter notebook
然后使用 jupyter notebook 命令打开Jupyter Notebook
浏览器会自动打开，进入 notebooks/ 目录，新建一个 Python Notebook (.ipynb 文件)，然后逐步运行代码。
注意我们用的kernel不是python3而是mbti_encoder，需要在Notebook的菜单栏中选择Kernel->Change Kernel->mbti_encoder。



#   M B T I _ T r a n s f o r m e r  
 #   M B T I _ T r a n s f o r m e r  
 