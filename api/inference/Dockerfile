FROM tensorflow/tensorflow:2.13.0

WORKDIR /app  
  
# Copy the requirements file to use for pip installations  
COPY requirements.txt .  
  
# RUN apt-get update && apt-get install -y \  
#     libgl1-mesa-glx \  
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt  
RUN pip install --no-cache-dir -r requirements.txt
  
# Copy your application files into the container  
COPY app.py .
COPY helper.py .  

RUN mkdir model_data
COPY model_data/articles_embed_df.pkl ./model_data/
COPY model_data/content_based_model.h5 ./model_data/
COPY model_data/df_articles.pkl ./model_data/
COPY model_data/embeddings_dict.pkl ./model_data/
COPY model_data/user_profiles_df_all.pkl ./model_data/

# Set the entrypoint for the container  
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080",  "--log-level", "debug", "app:app"]  
