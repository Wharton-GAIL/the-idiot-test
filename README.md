# The Idiot Test

![The Idiot Test Header](idiot-test-header-image.png)

An interactive web application to compare, analyze, and visualize the performance of different prompts using OpenAI and Anthropic models.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
    - [For macOS Users](#for-macos-users)
    - [For Windows Users](#for-windows-users)
- [Usage](#usage)
  - [Running the App](#running-the-app)
  - [Application Overview](#application-overview)

## Introduction

**The Idiot Test** is a tool designed for AI researchers, developers, and enthusiasts to evaluate and compare the effectiveness of different prompts when interacting with large language models like OpenAI's GPT series or Anthropic's models.

By running multiple iterations of prompts and analyzing the responses, users can gain statistical insights into how changes in prompts affect model outputs, including response length, ratings, and overall performance.

## Features

- **Side-by-Side Prompt Comparison**: Input control and experimental prompts to compare their performance directly.
- **Statistical Analysis**: Run multiple iterations and receive statistical metrics like average rating, median rating, and standard deviation.
- **Customizable Settings**: Choose from various models, adjust temperatures, and select the number of iterations for in-depth analysis.
- **Visualizations**: Generate and view charts displaying response length distributions and rating distributions.
- **API Key Management**: Securely input and store your OpenAI and Anthropic API keys within the application.
- **Downloadable Reports**: Export the analysis results as an HTML report for further review or sharing.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and interactive user experience.

## Installation

### Setup Instructions

#### For macOS Users

1. **Install Homebrew (if not already installed)**

   Homebrew makes it easy to install and manage software on macOS.

   Open the **Terminal** app (You can find it in Applications > Utilities or search using Spotlight) and run:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3 (if not already installed)**

   ```bash
   brew install python
   ```

3. **Install Git (if not already installed)**

   ```bash
   brew install git
   ```

4. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/the-idiot-test.git
   cd the-idiot-test
   ```

   *If you don't have Git or prefer not to use it, you can download the ZIP file from the repository's webpage and extract it.*

5. **Create a Virtual Environment (Optional but Recommended)**

   Creating a virtual environment keeps your project dependencies isolated.

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

6. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

#### For Windows Users

1. **Install Python 3**

   Download and install Python from the [official website](https://www.python.org/downloads/windows/).

   - **IMPORTANT**: During installation, check the option **"Add Python to PATH"**.

2. **Install Git (if not already installed)**

   Download and install Git from the [official website](https://git-scm.com/download/win).

3. **Open Command Prompt**

   Press `Win + R`, type `cmd`, and press Enter.

4. **Navigate to Your Desired Directory**

   Use `cd` to change directories. For example:

   ```bash
   cd C:\Users\YourUsername\Documents
   ```

5. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/the-idiot-test.git
   cd the-idiot-test
   ```

   *If you don't have Git or prefer not to use it, you can download the ZIP file from the repository's webpage and extract it.*

6. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

7. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the App

1. **Ensure You Are in the Project Directory**

   ```bash
   cd the-idiot-test
   ```

2. **Activate the Virtual Environment (If You Created One)**

   - **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

   - **Windows**:

     ```bash
     venv\Scripts\activate
     ```

3. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

4. **Access the App**

   Once the server is running, you'll see a URL in the terminal similar to `http://localhost:8501`. Open this URL in your web browser to access the app.

### Application Overview

#### 1. **API Key Configuration**

- On the sidebar, expand the **API Keys** section.
- Input your **OpenAI API Key** and/or **Anthropic API Key**.
- Click the **Save API Keys** button to store them securely in your browser. Note: saving keys does not work reliably between sessions.

#### 2. **Settings**

- **Number of Iterations**: Select how many times each prompt should be run for statistical significance (1 to 50).
- **Model for Response Generation**: Choose the AI model to generate responses (e.g., `gpt-4o`, `o1-preview`).
- **Temperature for Response Generation**: Adjust the randomness of responses; higher values yield more variation with each response. This should probably stay at 1.0.
- **Model for Rating**: Select the model used to rate the responses.
- **Temperature for Rating**: Adjust the randomness for the rating generation. This should probably stay at 0.0.

#### 3. **Analysis Options**

- **Analyze Length of Response**: If this is on, then the evaluation will include the number of characters in the response. This is useful if your test evaluates how long the AI response is, for example if you're trying to persuade it to give long answers.
- **Add a Table of All Responses**: Toggle on to display a table with all raw responses at the end of the analysis - the actual text the AI returns.

#### 4. **Input Prompts**

- **Control Message**: Input the initial prompt(s) you want to test. You can add multiple message pairs by clicking the **Add Message Pair** button.
- **Experimental Message**: Input the alternative prompt(s) for comparison.

#### 5. **Evaluation Rubrics**

- Provide custom rating prompt templates for both control and experimental messages.
- These templates should include `{response}` where the actual model response will be inserted.
- They must instruct the AI to ask for a rating in brackets like `[0]`. Otherwise, there will be errors as the software will not be able to find the rating. You can use any scale, for example from [0] to [10].