# The Idiot Test

![The Idiot Test Header](header-image.png)

An interactive web application to compare, analyze, and visualize the performance of different prompts using OpenAI and Anthropic models.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [For macOS Users](#for-macos-users)
  - [For Windows Users](#for-windows-users)
- [Usage](#usage)
  - [Running the App](#running-the-app)
  - [Application Overview](#application-overview)
- [Troubleshooting](#troubleshooting)
- [Getting Help](#getting-help)

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

## Prerequisites

Before you begin, ensure you have the following:

- A computer running **macOS** or **Windows**.
- An internet connection to download required software and access the application.
- **OpenAI API Key** and/or **Anthropic API Key**:
  - **OpenAI**: Sign up and get your API key from the [OpenAI Dashboard](https://platform.openai.com/account/api-keys).
  - **Anthropic**: Request access and obtain your API key from [Anthropic's website](https://www.anthropic.com).

## Installation

Follow the instructions below based on your operating system. If you've never written code or used a terminal/command prompt before, don't worry—we'll guide you through each step.

### 🍎 For macOS Users

1. **Open Terminal**
   - Click on the **Spotlight Search** (🔍) in the top-right corner or press `Command (⌘) + Space`.
   - Type `Terminal` and press **Enter**.

2. **Set Up GitHub Authentication**

   a. **Create a Personal Access Token (PAT)**:

   - Go to [GitHub Settings > Developer Settings > Personal Access Tokens > Tokens (classic)](https://github.com/settings/tokens)
   - Click **"Generate new token (classic)"**
   - Give it a name like **"Idiot Test App"**
   - Set expiration to **"No Expiration"**
   - Check the box next to **"repo"**
   - Click **"Generate token"** at the bottom
   - **IMPORTANT**: Copy the token immediately and save it somewhere safe—you won't be able to see it again! Make sure not to accidentally copy a space or extra characters.

   b. **Clone the Repository**:

   ```bash
   git clone https://github.com/danshapiro/the-idiot-test.git
   ```

   When prompted:

   - **Username**: Enter your GitHub username
   - **Password**: Paste the Personal Access Token you just created—**NOT** your GitHub password

   c. **Navigate to the Project Folder**:

   ```bash
   cd the-idiot-test
   ```

3. **Install Homebrew (Package Manager)**

   a. **Install Homebrew**:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   This may require you to enter your password. Use the password for your Mac.

   b. **Follow Any On-Screen Instructions** to complete the installation.

4. **Install Python 3**

   ```bash
   brew install python
   ```

5. **Create a Virtual Environment**

   A virtual environment keeps project dependencies isolated.

   ```bash
   python3 -m venv venv && source venv/bin/activate
   ```

   After activation, your Terminal prompt will begin with `(venv)`.

6. **Install Dependencies**

   a. **Upgrade pip**:

   ```bash
   pip install --upgrade pip
   ```

   b. **Install Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

### 🪟 For Windows Users

1. **Install Git for Windows**

   Git allows you to clone repositories and manage version control.

   a. **Download Git**:

   - Visit the [Git for Windows website](https://git-scm.com/download/win).
   - Download the latest installer.

   b. **Install Git**:

   - Run the downloaded installer.
   - Use the **default settings** throughout the installation.
   - **Note**: Ensure that the option **"Git Bash Here"** is enabled during installation.

2. **Open Git Bash**

   - After installation, right-click on your desktop or inside any folder.
   - Select **"Git Bash Here"** from the context menu.
   - This will open a terminal window similar to the macOS Terminal.

3. **Set Up GitHub Authentication**

   a. **Create a Personal Access Token (PAT)**:

   - Go to [GitHub Settings > Developer Settings > Personal Access Tokens > Tokens (classic)](https://github.com/settings/tokens)
   - Click **"Generate new token (classic)"**
   - Give it a name like **"Idiot Test App"**
   - Set expiration to **"No Expiration"**
   - Check the box next to **"repo"**
   - Click **"Generate token"** at the bottom
   - **IMPORTANT**: Copy the token immediately and save it somewhere safe—you won't be able to see it again!

   b. **Configure Git with Your GitHub Credentials**:

   ```bash
   git config --global user.name "YourGitHubUsername"
   git config --global user.email "YourEmail@example.com"
   ```

4. **Clone the Repository**

   a. **Navigate to Your Desired Directory** (optional):

   - By default, Git Bash starts in your home directory.
   - To change the directory, use the `cd` command. For example:

     ```bash
     cd /c/Users/YourUsername/Documents
     ```

     Replace `YourUsername` with your actual Windows username.

   b. **Clone the Repository**:

   ```bash
   git clone https://github.com/danshapiro/the-idiot-test.git
   ```

   When prompted:

   - **Username**: Enter your GitHub username
   - **Password**: Paste your Personal Access Token—**NOT** your GitHub password

   c. **Navigate to the Project Folder**:

   ```bash
   cd the-idiot-test
   ```

5. **Install Python 3**

   a. **Verify if Python is Already Installed**:

   ```bash
   python --version
   ```

   - If Python 3.x is installed, you can proceed to the next step.
   - If not, continue with the installation.

   b. **Download Python**:

   - Visit the [official Python website](https://www.python.org/downloads/windows/).
   - Download the latest **Python 3** installer for Windows.

   c. **Install Python**:

   - Run the downloaded installer.
   - **Important**: During installation, check the box that says **"Add Python to PATH"**.
   - Follow the on-screen instructions to complete the installation.

   d. **Verify Installation**:

   - Close and reopen **Git Bash**.
   - Type:

     ```bash
     python --version
     ```

     This should display the Python version installed.

6. **Create a Virtual Environment**

   A virtual environment keeps project dependencies isolated.

   a. **Create the Virtual Environment**:

   ```bash
   python -m venv venv
   ```

   b. **Activate the Virtual Environment**:

   ```bash
   source venv/Scripts/activate
   ```

   - After activation, your Git Bash prompt will begin with `(venv)`.

7. **Install Dependencies**

   a. **Upgrade pip**:

   ```bash
   pip install --upgrade pip
   ```

   b. **Install Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the App

#### 1. Ensure You Are in the Project Directory

  ```bash
  cd path/to/the-idiot-test
  ```

  Replace `path/to` with the actual path to the project folder.

#### 2. Activate the Virtual Environment

- **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

- **Windows** (using Git Bash):

  ```bash
  source venv/Scripts/activate
  ```

  After activation, your prompt will begin with `(venv)`.

#### 3. Run the Streamlit App

- **Start the Application**:

  ```bash
  streamlit run app.py
  ```

- **Note**: If you receive an error saying `streamlit: command not found` or similar, ensure that the virtual environment is activated and that the dependencies were installed correctly.

#### 4. Access the App

- **Open your web browser**.
- **Navigate to** the URL displayed in the terminal, typically:

  ```
  http://localhost:8501
  ```

  If the app doesn't open automatically, copy and paste the URL into your browser.

### Application Overview

#### 1. API Key Configuration

Before using the app, you'll need to enter your API keys.

- **Open the Sidebar**:
  - Click on the arrow `>` in the top-left corner of the app.
- **Expand the "API Keys" Section**.
- **Input Your API Keys**:
  - **OpenAI API Key**:
    - Paste your key into the provided field.
  - **Anthropic API Key** (if you have one):
    - Paste your key into the provided field.
- **Save the API Keys**:
  - Click the **"Save API Keys"** button.
- **Note**: The keys are stored locally in your browser for this session. You may need to re-enter them the next time you run the app.

#### 2. Settings

Adjust the settings to customize your analysis.

- **Number of Iterations**:
  - Choose how many times each prompt should be run (1 to 50).
  - More iterations provide better statistical significance but take longer.

- **Model for Response Generation**:
  - Select the AI model for generating responses (e.g., `gpt-4`, `gpt-3.5-turbo`).
  - **Note**: Availability of models depends on your API access.

- **Temperature for Response Generation**:
  - Controls the randomness of the responses.
  - Values range from `0.0` (deterministic) to `1.0` (most random).
  - Recommended to keep at `1.0` for diverse responses.

- **Model for Rating**:
  - Choose the AI model used to rate the responses.
  - Typically a faster, less expensive model is sufficient.

- **Temperature for Rating**:
  - Recommended to keep at `0.0` for consistent ratings.

#### 3. Analysis Options

- **Analyze Length of Response**:
  - Toggle **ON** to include response length in the evaluation.
  - Useful for testing prompts that aim for longer or shorter responses.

- **Add a Table of All Responses**:
  - Toggle **ON** to display a table of all raw responses.
  - Helps in reviewing each individual response.

#### 4. Input Prompts

Set up the prompts you wish to test.

- **Control Message**:
  - Enter your primary prompt in the **"Control Message"** field.
  - You can add multiple message pairs if needed.

- **Experimental Message**:
  - Enter the alternative prompt you want to compare against the control.

- **Add Message Pair**:
  - Click this button to add more message pairs if your prompts involve back-and-forth interaction.

#### 5. Evaluation Rubrics

Customize how the responses are evaluated.

- **Rating Prompt Template for Control Messages**:
  - Provide a rubric or set of instructions that tells the AI how to rate the responses.
  - **Example**:

    ```
    Evaluate the response for clarity and completeness. Provide a rating between [0] (poor) and [10] (excellent).
    ```

- **Rating Prompt Template for Experimental Messages**:
  - Similar to the control, but can be adjusted if you want different evaluation criteria.

- **Important**:
  - Include `{response}` in your templates where you want the AI's response to be inserted.
  - Ensure the AI is instructed to provide a rating enclosed in brackets, like `[7]`.

#### 6. Run the Analysis

- **Start the Evaluation**:
  - Click the **"Run Analysis"** button at the bottom of the page.
- **Wait for Completion**:
  - The app will display progress bars and status messages.
  - Processing time depends on the number of iterations and model speeds.

#### 7. View Results

- **Statistical Summary**:
  - Average rating, median rating, standard deviation, etc.

- **Charts and Graphs**:
  - Visual representations of response lengths and ratings distributions.

- **Response Table** (if enabled):
  - A table listing all individual responses and their respective ratings.

#### 8. Exporting Results

- **Download Report**:
  - Click the **"Download Report"** button to save the analysis as an HTML file.
  - This report can be shared or viewed later without running the app again.

## Troubleshooting

- **Streamlit Command Not Found**:
  - Ensure the virtual environment is activated.
  - Verify that `streamlit` is installed by running `pip show streamlit`.

- **API Errors**:
  - Double-check your API keys.
  - Ensure you have enough quota or credits with the API provider.

- **Module Not Found Errors**:
  - Install missing packages:

    ```bash
    pip install package-name
    ```

- **General Issues**:
  - Make sure all steps were followed correctly.
  - Search for error messages online or consult the project repository for issues.

## Getting Help

If you encounter any problems or have questions:

- **Check the GitHub Issues Page**: See if others have had similar issues.
- **Open a New Issue**: Provide details about the problem for assistance.
- **Contact the Developer**: Reach out via the contact information provided in the repository.

Happy experimenting with prompts!