# Buzz 📰

## Project Overview

This project aims to predict the topic of a given news article using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA). It also compares the performance of both algorithms using t-Distributed Stochastic Neighbor Embedding (TSNE). The project leverages the News3K library to fetch news articles and utilizes Flask for the web interface. To ensure efficiency, multiprocessing is used to run each model as a separate process.

## Features

- **Topic Prediction**: Predicts the topic of a given news article using LSA and LDA.
- **Performance Comparison**: Uses TSNE to compare the performance of LSA and LDA.
- **News Fetching**: Utilizes the News3K library to fetch the latest news articles.
- **Web Interface**: Simple HTML template to interact with the Flask server.
- **Multiprocessing**: Runs each model as a separate process to enhance efficiency.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/news-topic-prediction.git
    cd news-topic-prediction
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask server**:
    ```sh
    python app.py
    ```

2. **Access the web interface**:
    Open your web browser and go to `http://127.0.0.1:5000`.

3. **Predict the topic**:
    - Enter the news article text in the provided input field.
    - Click the "Predict Topic" button to get the predicted topic using both LSA and LDA.
    - View the TSNE visualization to compare the performance of the two models.

## Project Structure

- `app.py`: The main Flask application.
- `models.py`: Contains the LSA and LDA model implementations.
- `news_fetcher.py`: Uses the News3K library to fetch news articles.
- `templates/`: Directory containing the HTML template for the web interface.
- `static/`: Directory for static files (e.g., CSS, JavaScript).
- `requirements.txt`: List of required Python packages.

## Dependencies

- Flask
- News3K
- scikit-learn
- gensim
- numpy
- pandas
- matplotlib
- seaborn
- TSNE
- multiprocessing

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
