{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled6.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "[View in Colaboratory](https://colab.research.google.com/github/meghamattikalli/internship/blob/master/working13.py)"
      ]
    },
    {
      "metadata": {
        "id": "YTHkBSOfwpMi",
        "colab_type": "code",
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7Ci8vIE1heCBhbW91bnQgb2YgdGltZSB0byBibG9jayB3YWl0aW5nIGZvciB0aGUgdXNlci4KY29uc3QgRklMRV9DSEFOR0VfVElNRU9VVF9NUyA9IDMwICogMTAwMDsKCmZ1bmN0aW9uIF91cGxvYWRGaWxlcyhpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IHN0ZXBzID0gdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKTsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIC8vIENhY2hlIHN0ZXBzIG9uIHRoZSBvdXRwdXRFbGVtZW50IHRvIG1ha2UgaXQgYXZhaWxhYmxlIGZvciB0aGUgbmV4dCBjYWxsCiAgLy8gdG8gdXBsb2FkRmlsZXNDb250aW51ZSBmcm9tIFB5dGhvbi4KICBvdXRwdXRFbGVtZW50LnN0ZXBzID0gc3RlcHM7CgogIHJldHVybiBfdXBsb2FkRmlsZXNDb250aW51ZShvdXRwdXRJZCk7Cn0KCi8vIFRoaXMgaXMgcm91Z2hseSBhbiBhc3luYyBnZW5lcmF0b3IgKG5vdCBzdXBwb3J0ZWQgaW4gdGhlIGJyb3dzZXIgeWV0KSwKLy8gd2hlcmUgdGhlcmUgYXJlIG11bHRpcGxlIGFzeW5jaHJvbm91cyBzdGVwcyBhbmQgdGhlIFB5dGhvbiBzaWRlIGlzIGdvaW5nCi8vIHRvIHBvbGwgZm9yIGNvbXBsZXRpb24gb2YgZWFjaCBzdGVwLgovLyBUaGlzIHVzZXMgYSBQcm9taXNlIHRvIGJsb2NrIHRoZSBweXRob24gc2lkZSBvbiBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcCwKLy8gdGhlbiBwYXNzZXMgdGhlIHJlc3VsdCBvZiB0aGUgcHJldmlvdXMgc3RlcCBhcyB0aGUgaW5wdXQgdG8gdGhlIG5leHQgc3RlcC4KZnVuY3Rpb24gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpIHsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIGNvbnN0IHN0ZXBzID0gb3V0cHV0RWxlbWVudC5zdGVwczsKCiAgY29uc3QgbmV4dCA9IHN0ZXBzLm5leHQob3V0cHV0RWxlbWVudC5sYXN0UHJvbWlzZVZhbHVlKTsKICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5leHQudmFsdWUucHJvbWlzZSkudGhlbigodmFsdWUpID0+IHsKICAgIC8vIENhY2hlIHRoZSBsYXN0IHByb21pc2UgdmFsdWUgdG8gbWFrZSBpdCBhdmFpbGFibGUgdG8gdGhlIG5leHQKICAgIC8vIHN0ZXAgb2YgdGhlIGdlbmVyYXRvci4KICAgIG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSA9IHZhbHVlOwogICAgcmV0dXJuIG5leHQudmFsdWUucmVzcG9uc2U7CiAgfSk7Cn0KCi8qKgogKiBHZW5lcmF0b3IgZnVuY3Rpb24gd2hpY2ggaXMgY2FsbGVkIGJldHdlZW4gZWFjaCBhc3luYyBzdGVwIG9mIHRoZSB1cGxvYWQKICogcHJvY2Vzcy4KICogQHBhcmFtIHtzdHJpbmd9IGlucHV0SWQgRWxlbWVudCBJRCBvZiB0aGUgaW5wdXQgZmlsZSBwaWNrZXIgZWxlbWVudC4KICogQHBhcmFtIHtzdHJpbmd9IG91dHB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIG91dHB1dCBkaXNwbGF5LgogKiBAcmV0dXJuIHshSXRlcmFibGU8IU9iamVjdD59IEl0ZXJhYmxlIG9mIG5leHQgc3RlcHMuCiAqLwpmdW5jdGlvbiogdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKSB7CiAgY29uc3QgaW5wdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaW5wdXRJZCk7CiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gZmFsc2U7CgogIGNvbnN0IG91dHB1dEVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChvdXRwdXRJZCk7CiAgb3V0cHV0RWxlbWVudC5pbm5lckhUTUwgPSAnJzsKCiAgY29uc3QgcGlja2VkUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBpbnB1dEVsZW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKGUpID0+IHsKICAgICAgcmVzb2x2ZShlLnRhcmdldC5maWxlcyk7CiAgICB9KTsKICB9KTsKCiAgY29uc3QgY2FuY2VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgaW5wdXRFbGVtZW50LnBhcmVudEVsZW1lbnQuYXBwZW5kQ2hpbGQoY2FuY2VsKTsKICBjYW5jZWwudGV4dENvbnRlbnQgPSAnQ2FuY2VsIHVwbG9hZCc7CiAgY29uc3QgY2FuY2VsUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBjYW5jZWwub25jbGljayA9ICgpID0+IHsKICAgICAgcmVzb2x2ZShudWxsKTsKICAgIH07CiAgfSk7CgogIC8vIENhbmNlbCB1cGxvYWQgaWYgdXNlciBoYXNuJ3QgcGlja2VkIGFueXRoaW5nIGluIHRpbWVvdXQuCiAgY29uc3QgdGltZW91dFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9LCBGSUxFX0NIQU5HRV9USU1FT1VUX01TKTsKICB9KTsKCiAgLy8gV2FpdCBmb3IgdGhlIHVzZXIgdG8gcGljayB0aGUgZmlsZXMuCiAgY29uc3QgZmlsZXMgPSB5aWVsZCB7CiAgICBwcm9taXNlOiBQcm9taXNlLnJhY2UoW3BpY2tlZFByb21pc2UsIHRpbWVvdXRQcm9taXNlLCBjYW5jZWxQcm9taXNlXSksCiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdzdGFydGluZycsCiAgICB9CiAgfTsKCiAgaWYgKCFmaWxlcykgewogICAgcmV0dXJuIHsKICAgICAgcmVzcG9uc2U6IHsKICAgICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICAgIH0KICAgIH07CiAgfQoKICBjYW5jZWwucmVtb3ZlKCk7CgogIC8vIERpc2FibGUgdGhlIGlucHV0IGVsZW1lbnQgc2luY2UgZnVydGhlciBwaWNrcyBhcmUgbm90IGFsbG93ZWQuCiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gdHJ1ZTsKCiAgZm9yIChjb25zdCBmaWxlIG9mIGZpbGVzKSB7CiAgICBjb25zdCBsaSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2xpJyk7CiAgICBsaS5hcHBlbmQoc3BhbihmaWxlLm5hbWUsIHtmb250V2VpZ2h0OiAnYm9sZCd9KSk7CiAgICBsaS5hcHBlbmQoc3BhbigKICAgICAgICBgKCR7ZmlsZS50eXBlIHx8ICduL2EnfSkgLSAke2ZpbGUuc2l6ZX0gYnl0ZXMsIGAgKwogICAgICAgIGBsYXN0IG1vZGlmaWVkOiAkewogICAgICAgICAgICBmaWxlLmxhc3RNb2RpZmllZERhdGUgPyBmaWxlLmxhc3RNb2RpZmllZERhdGUudG9Mb2NhbGVEYXRlU3RyaW5nKCkgOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnbi9hJ30gLSBgKSk7CiAgICBjb25zdCBwZXJjZW50ID0gc3BhbignMCUgZG9uZScpOwogICAgbGkuYXBwZW5kQ2hpbGQocGVyY2VudCk7CgogICAgb3V0cHV0RWxlbWVudC5hcHBlbmRDaGlsZChsaSk7CgogICAgY29uc3QgZmlsZURhdGFQcm9taXNlID0gbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHsKICAgICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgcmVhZGVyLm9ubG9hZCA9IChlKSA9PiB7CiAgICAgICAgcmVzb2x2ZShlLnRhcmdldC5yZXN1bHQpOwogICAgICB9OwogICAgICByZWFkZXIucmVhZEFzQXJyYXlCdWZmZXIoZmlsZSk7CiAgICB9KTsKICAgIC8vIFdhaXQgZm9yIHRoZSBkYXRhIHRvIGJlIHJlYWR5LgogICAgbGV0IGZpbGVEYXRhID0geWllbGQgewogICAgICBwcm9taXNlOiBmaWxlRGF0YVByb21pc2UsCiAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgYWN0aW9uOiAnY29udGludWUnLAogICAgICB9CiAgICB9OwoKICAgIC8vIFVzZSBhIGNodW5rZWQgc2VuZGluZyB0byBhdm9pZCBtZXNzYWdlIHNpemUgbGltaXRzLiBTZWUgYi82MjExNTY2MC4KICAgIGxldCBwb3NpdGlvbiA9IDA7CiAgICB3aGlsZSAocG9zaXRpb24gPCBmaWxlRGF0YS5ieXRlTGVuZ3RoKSB7CiAgICAgIGNvbnN0IGxlbmd0aCA9IE1hdGgubWluKGZpbGVEYXRhLmJ5dGVMZW5ndGggLSBwb3NpdGlvbiwgTUFYX1BBWUxPQURfU0laRSk7CiAgICAgIGNvbnN0IGNodW5rID0gbmV3IFVpbnQ4QXJyYXkoZmlsZURhdGEsIHBvc2l0aW9uLCBsZW5ndGgpOwogICAgICBwb3NpdGlvbiArPSBsZW5ndGg7CgogICAgICBjb25zdCBiYXNlNjQgPSBidG9hKFN0cmluZy5mcm9tQ2hhckNvZGUuYXBwbHkobnVsbCwgY2h1bmspKTsKICAgICAgeWllbGQgewogICAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgICBhY3Rpb246ICdhcHBlbmQnLAogICAgICAgICAgZmlsZTogZmlsZS5uYW1lLAogICAgICAgICAgZGF0YTogYmFzZTY0LAogICAgICAgIH0sCiAgICAgIH07CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPQogICAgICAgICAgYCR7TWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCl9JSBkb25lYDsKICAgIH0KICB9CgogIC8vIEFsbCBkb25lLgogIHlpZWxkIHsKICAgIHJlc3BvbnNlOiB7CiAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgIH0KICB9Owp9CgpzY29wZS5nb29nbGUgPSBzY29wZS5nb29nbGUgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYiA9IHNjb3BlLmdvb2dsZS5jb2xhYiB8fCB7fTsKc2NvcGUuZ29vZ2xlLmNvbGFiLl9maWxlcyA9IHsKICBfdXBsb2FkRmlsZXMsCiAgX3VwbG9hZEZpbGVzQ29udGludWUsCn07Cn0pKHNlbGYpOwo=",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 142
        },
        "outputId": "7d4766ff-968b-4b00-acad-1374b06e2e9a"
      },
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()\n"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-1543f5f1-83c0-4926-a526-ce089ec9efc1\" name=\"files[]\" multiple disabled />\n",
              "     <output id=\"result-1543f5f1-83c0-4926-a526-ce089ec9efc1\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Saving five_thousand.csv to five_thousand (2).csv\n",
            "Saving Indian-Female-Names1.csv to Indian-Female-Names1 (2).csv\n",
            "Saving Indian-Male-Names1.csv to Indian-Male-Names1 (2).csv\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "TNzKw9b5wjph",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 4072
        },
        "outputId": "1c44c837-80d7-4c60-faa6-bbcaa5c69347"
      },
      "cell_type": "code",
      "source": [
        "# -*- coding: utf-8 -*-\n",
        "\"\"\"\n",
        "Created on Tue Aug 14 14:07:06 2018\n",
        "accuracy is 0.908\n",
        "roc-auc is 0.909\n",
        "3 hidden layers with 100,50,10 nodes respectively\n",
        "@author: HP\n",
        "\"\"\"\n",
        "\n",
        "# -*- coding: utf-8 -*-\n",
        "\"\"\"\n",
        "Created on Tue Aug 14 11:53:53 2018\n",
        "accuracy is 0.889\n",
        "roc-auc is 0.889\n",
        "3 hidden layers with 6 nodes each\n",
        "Names with less than 2 letters removed\n",
        "10000 samples\n",
        "\n",
        "@author: HP\n",
        "\"\"\"\n",
        "\n",
        "import pandas as pd\n",
        "import io\n",
        "\n",
        "female = pd.read_csv(io.StringIO(uploaded['Indian-Female-Names1.csv'].decode('utf-8')))\n",
        "\n",
        "male = pd.read_csv(io.StringIO(uploaded['Indian-Male-Names1.csv'].decode('utf-8')))\n",
        "\n",
        "eng_word = pd.read_csv(io.StringIO(uploaded['five_thousand.csv'].decode('utf-8')))\n",
        "\n",
        "# Importing the dataset\n",
        "#female= pd.read_csv('Indian-Female-Names1.csv')\n",
        "female= female.apply(lambda x: x.astype(str).str.lower())\n",
        "female=female.drop_duplicates(subset=['word'], keep='first')\n",
        "female=female.iloc[:,0:1]\n",
        "female[\"type\"]=1\n",
        "\n",
        "unwanted=['mr','.','moh.','@','miss','&','mrs','/','ku.','km',',','km0','-','(','smt','smt.','`','[','ms']\n",
        "for i in unwanted:\n",
        "    female['word'] = female.word.str.replace(i, '')\n",
        "numbers=['1','2','3','4','5','6','7','8','9','0']\n",
        "for i in numbers:\n",
        "    female['word'] = female.word.str.replace(i, '')\n",
        "female['word'] = female.word.str.strip()\n",
        "female['word'] = female.word.str.partition(\" \")\n",
        "female=female.drop_duplicates(subset=['word'], keep='first')\n",
        "\n",
        "#preprocessing of names\n",
        "#female['word'] = female.word.str.replace('miss', '')\n",
        "#female['word'] = female.word.str.replace('mrs', '') \n",
        "#female['word'] = female.word.str.replace('.', '')\n",
        "#female['word'] = female.word.str.replace('smt', '')\n",
        "#female['word'] = female.word.str.replace('@', '')\n",
        "#female['word'] = female.word.str.strip()\n",
        "#female['word'] = female.word.str.partition(\" \")\n",
        "#female=female.drop_duplicates(subset=['word'], keep='first')\n",
        "female=female[female['word'].map(len) > 2]\n",
        "female=female.head(2500)\n",
        "\n",
        "\n",
        "#male= pd.read_csv('Indian-Male-Names1.csv')\n",
        "male= male.apply(lambda x: x.astype(str).str.lower())\n",
        "male= male.drop_duplicates(subset=['word'], keep='first')\n",
        "male=male.iloc[:,0:1]\n",
        "male[\"type\"]=1\n",
        "\n",
        "for i in unwanted:\n",
        "    male['word'] = male.word.str.replace(i, '')\n",
        "for i in numbers:\n",
        "    male['word'] = male.word.str.replace(i, '')\n",
        "    \n",
        "male['word'] = male.word.str.strip()\n",
        "male['word'] = male.word.str.partition(\" \")\n",
        "male=male.drop_duplicates(subset=['word'], keep='first')\n",
        "\n",
        "#male['word'] = male.word.str.replace('mr', '')\n",
        "#male['word'] = male.word.str.replace('.', '')\n",
        "#male['name'] = male.name.str.replace('smt', '')\n",
        "#male['word'] = male.word.str.replace('@', '')\n",
        "#male['word'] = male.word.str.strip()\n",
        "#male['word'] = male.word.str.partition(\" \")\n",
        "male=male[male['word'].map(len) > 2]\n",
        "male=male.head(2500)\n",
        "\n",
        "\n",
        "#english words\n",
        "#eng_word = pd.read_csv('five_thousand.csv')\n",
        "eng_word = eng_word.apply(lambda x: x.astype(str).str.lower())\n",
        "eng_word['word'] = eng_word.word.str.replace('-', '')\n",
        "eng_word['word'] = eng_word.word.str.replace(\"'\", '')\n",
        "eng_word['word'] = eng_word.word.str.replace(\"/\", ' ')\n",
        "eng_word['word'] = eng_word.word.str.partition(\" \")\n",
        "\n",
        "\n",
        "#eng_pre = pd.read_csv('prepositions.csv')\n",
        "eng_word['type'] = 0\n",
        "#eng_pre['type'] = 0\n",
        "\n",
        "\n",
        "#concatenate names and english words\n",
        "names=pd.concat([female,male,eng_word], axis=0)\n",
        "names=names.dropna()\n",
        "\n",
        "\n",
        "names.index=[i for i in range(0,len(names.index))] #renaming the indices\n",
        "\n",
        "X=names['word'] #features\n",
        "y=names['type'] #labels\n",
        "        \n",
        "#padding and truncating X\n",
        "for i in range(0,len(X.index)):\n",
        "    if(len(X.loc[i])>10):\n",
        "        X.loc[i]=X.loc[i][0:10]\n",
        "    else:    \n",
        "        X.loc[i]=X.loc[i].ljust(10) \n",
        "\n",
        "def letter_to_int(letter):\n",
        "    alphabet = list('abcdefghijklmnopqrstuvwxyz ')\n",
        "    return alphabet.index(letter)\n",
        "\n",
        "\n",
        "list1=[]\n",
        "for j in range(0,len(X.index)):\n",
        "    z=[]\n",
        "    for i in X.loc[j]:\n",
        "        z.append(letter_to_int(i))\n",
        "    list1.append(tuple(z))\n",
        "        \n",
        "df=pd.DataFrame(list1)\n",
        "\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "onehotencoder = OneHotEncoder(categorical_features = [0])\n",
        "df1 = onehotencoder.fit_transform(df).toarray()\n",
        "df1 = df1[:, 1:]\n",
        "\n",
        "for i in range(9,0,-1):\n",
        "    zzz=len(df1[0])-i\n",
        "    onehotencoder = OneHotEncoder(categorical_features = [zzz])\n",
        "    df1 = onehotencoder.fit_transform(df1).toarray()\n",
        "    df1 = df1[:, 1:]\n",
        "\n",
        "# Splitting the dataset into the Training set and Test set\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size = 0.2, random_state = 0)\n",
        "\n",
        "\n",
        "# Part 2 - Now let's make the ANN!\n",
        "\n",
        "# Importing the Keras libraries and packages\n",
        "\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense\n",
        "\n",
        "# Initialising the ANN\n",
        "classifier = Sequential()\n",
        "\n",
        "# Adding the input layer and the first hidden layer\n",
        "classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim = len(df1[0])))\n",
        "\n",
        "# Adding the second hidden layer\n",
        "classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))\n",
        "\n",
        "# Adding the third hidden layer\n",
        "classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))\n",
        "\n",
        "# Adding the output layer\n",
        "classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))\n",
        "\n",
        "\n",
        "# Compiling the ANN\n",
        "classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])\n",
        "classifier.summary()\n",
        "\n",
        "# Fitting the ANN to the Training set\n",
        "classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)\n",
        "\n",
        "\n",
        "# Predicting the Test set results\n",
        "y_pred = classifier.predict(X_test)\n",
        "y_pred = (y_pred > 0.5)\n",
        "\n",
        "# Making the Confusion Matrix\n",
        "from sklearn.metrics import confusion_matrix\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "\n",
        "\n",
        "from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score\n",
        "# Print model performance and plot the roc curve\n",
        "print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred)))\n",
        "print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred)))"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/pandas/core/indexing.py:194: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy\n",
            "  self._setitem_with_indexer(indexer, value)\n",
            "Using TensorFlow backend.\n",
            "/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:157: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation=\"relu\", input_dim=251, units=100, kernel_initializer=\"uniform\")`\n",
            "/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:160: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation=\"relu\", units=50, kernel_initializer=\"uniform\")`\n",
            "/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:163: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation=\"relu\", units=10, kernel_initializer=\"uniform\")`\n",
            "/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:166: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation=\"sigmoid\", units=1, kernel_initializer=\"uniform\")`\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "dense_1 (Dense)              (None, 100)               25200     \n",
            "_________________________________________________________________\n",
            "dense_2 (Dense)              (None, 50)                5050      \n",
            "_________________________________________________________________\n",
            "dense_3 (Dense)              (None, 10)                510       \n",
            "_________________________________________________________________\n",
            "dense_4 (Dense)              (None, 1)                 11        \n",
            "=================================================================\n",
            "Total params: 30,771\n",
            "Trainable params: 30,771\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "Epoch 1/100\n",
            "8000/8000 [==============================] - 6s 786us/step - loss: 0.4318 - acc: 0.7910\n",
            "Epoch 2/100\n",
            "8000/8000 [==============================] - 5s 664us/step - loss: 0.2920 - acc: 0.8751\n",
            "Epoch 3/100\n",
            "8000/8000 [==============================] - 5s 662us/step - loss: 0.2259 - acc: 0.9102\n",
            "Epoch 4/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.1747 - acc: 0.9306\n",
            "Epoch 5/100\n",
            "8000/8000 [==============================] - 5s 650us/step - loss: 0.1314 - acc: 0.9504\n",
            "Epoch 6/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.1080 - acc: 0.9580\n",
            "Epoch 7/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0914 - acc: 0.9695\n",
            "Epoch 8/100\n",
            "8000/8000 [==============================] - 5s 661us/step - loss: 0.0708 - acc: 0.9752\n",
            "Epoch 9/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0633 - acc: 0.9776\n",
            "Epoch 10/100\n",
            "8000/8000 [==============================] - 5s 664us/step - loss: 0.0541 - acc: 0.9812\n",
            "Epoch 11/100\n",
            "8000/8000 [==============================] - 5s 663us/step - loss: 0.0431 - acc: 0.9842\n",
            "Epoch 12/100\n",
            "8000/8000 [==============================] - 5s 662us/step - loss: 0.0428 - acc: 0.9855\n",
            "Epoch 13/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0391 - acc: 0.9866\n",
            "Epoch 14/100\n",
            "8000/8000 [==============================] - 5s 666us/step - loss: 0.0365 - acc: 0.9876\n",
            "Epoch 15/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0316 - acc: 0.9892\n",
            "Epoch 16/100\n",
            "8000/8000 [==============================] - 5s 663us/step - loss: 0.0264 - acc: 0.9910\n",
            "Epoch 17/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0303 - acc: 0.9895\n",
            "Epoch 18/100\n",
            "8000/8000 [==============================] - 5s 654us/step - loss: 0.0290 - acc: 0.9895\n",
            "Epoch 19/100\n",
            "8000/8000 [==============================] - 5s 654us/step - loss: 0.0184 - acc: 0.9927\n",
            "Epoch 20/100\n",
            "8000/8000 [==============================] - 5s 662us/step - loss: 0.0238 - acc: 0.9910\n",
            "Epoch 21/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0177 - acc: 0.9930\n",
            "Epoch 22/100\n",
            "8000/8000 [==============================] - 5s 659us/step - loss: 0.0217 - acc: 0.9910\n",
            "Epoch 23/100\n",
            "8000/8000 [==============================] - 5s 659us/step - loss: 0.0224 - acc: 0.9909\n",
            "Epoch 24/100\n",
            "8000/8000 [==============================] - 5s 669us/step - loss: 0.0188 - acc: 0.9929\n",
            "Epoch 25/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0131 - acc: 0.9945\n",
            "Epoch 26/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0151 - acc: 0.9940\n",
            "Epoch 27/100\n",
            "8000/8000 [==============================] - 5s 659us/step - loss: 0.0149 - acc: 0.9932\n",
            "Epoch 28/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0132 - acc: 0.9941\n",
            "Epoch 29/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0174 - acc: 0.9934\n",
            "Epoch 30/100\n",
            "8000/8000 [==============================] - 5s 652us/step - loss: 0.0143 - acc: 0.9936\n",
            "Epoch 31/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0142 - acc: 0.9937\n",
            "Epoch 32/100\n",
            "8000/8000 [==============================] - 5s 654us/step - loss: 0.0076 - acc: 0.9962\n",
            "Epoch 33/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0185 - acc: 0.9929\n",
            "Epoch 34/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0140 - acc: 0.9940\n",
            "Epoch 35/100\n",
            "8000/8000 [==============================] - 5s 653us/step - loss: 0.0186 - acc: 0.9935\n",
            "Epoch 36/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0097 - acc: 0.9957\n",
            "Epoch 37/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0064 - acc: 0.9971\n",
            "Epoch 38/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0066 - acc: 0.9967\n",
            "Epoch 39/100\n",
            "8000/8000 [==============================] - 5s 661us/step - loss: 0.0195 - acc: 0.9931\n",
            "Epoch 40/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0103 - acc: 0.9954\n",
            "Epoch 41/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0113 - acc: 0.9956\n",
            "Epoch 42/100\n",
            "8000/8000 [==============================] - 5s 654us/step - loss: 0.0094 - acc: 0.9956\n",
            "Epoch 43/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0124 - acc: 0.9942\n",
            "Epoch 44/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0071 - acc: 0.9969\n",
            "Epoch 45/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0065 - acc: 0.9969\n",
            "Epoch 46/100\n",
            "8000/8000 [==============================] - 5s 669us/step - loss: 0.0144 - acc: 0.9944\n",
            "Epoch 47/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0123 - acc: 0.9952\n",
            "Epoch 48/100\n",
            "8000/8000 [==============================] - 5s 659us/step - loss: 0.0142 - acc: 0.9947\n",
            "Epoch 49/100\n",
            "8000/8000 [==============================] - 5s 653us/step - loss: 0.0097 - acc: 0.9965\n",
            "Epoch 50/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0122 - acc: 0.9955\n",
            "Epoch 51/100\n",
            "8000/8000 [==============================] - 5s 652us/step - loss: 0.0062 - acc: 0.9969\n",
            "Epoch 52/100\n",
            "8000/8000 [==============================] - 5s 654us/step - loss: 0.0051 - acc: 0.9975\n",
            "Epoch 53/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0043 - acc: 0.9975\n",
            "Epoch 54/100\n",
            "8000/8000 [==============================] - 5s 673us/step - loss: 0.0042 - acc: 0.9976\n",
            "Epoch 55/100\n",
            "8000/8000 [==============================] - 5s 670us/step - loss: 0.0138 - acc: 0.9932\n",
            "Epoch 56/100\n",
            "8000/8000 [==============================] - 5s 681us/step - loss: 0.0111 - acc: 0.9962\n",
            "Epoch 57/100\n",
            "8000/8000 [==============================] - 5s 662us/step - loss: 0.0053 - acc: 0.9969\n",
            "Epoch 58/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0102 - acc: 0.9960\n",
            "Epoch 59/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0116 - acc: 0.9955\n",
            "Epoch 60/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0070 - acc: 0.9972\n",
            "Epoch 61/100\n",
            "8000/8000 [==============================] - 5s 653us/step - loss: 0.0044 - acc: 0.9974\n",
            "Epoch 62/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0103 - acc: 0.9952\n",
            "Epoch 63/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0054 - acc: 0.9975\n",
            "Epoch 64/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0040 - acc: 0.9976\n",
            "Epoch 65/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0037 - acc: 0.9974\n",
            "Epoch 66/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0129 - acc: 0.9956\n",
            "Epoch 67/100\n",
            "8000/8000 [==============================] - 5s 660us/step - loss: 0.0081 - acc: 0.9965\n",
            "Epoch 68/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0119 - acc: 0.9944\n",
            "Epoch 69/100\n",
            "8000/8000 [==============================] - 5s 665us/step - loss: 0.0056 - acc: 0.9970\n",
            "Epoch 70/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0062 - acc: 0.9969\n",
            "Epoch 71/100\n",
            "8000/8000 [==============================] - 5s 661us/step - loss: 0.0055 - acc: 0.9971\n",
            "Epoch 72/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0112 - acc: 0.9964\n",
            "Epoch 73/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0072 - acc: 0.9965\n",
            "Epoch 74/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0042 - acc: 0.9975\n",
            "Epoch 75/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0071 - acc: 0.9961\n",
            "Epoch 76/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0168 - acc: 0.9946\n",
            "Epoch 77/100\n",
            "8000/8000 [==============================] - 5s 660us/step - loss: 0.0072 - acc: 0.9965\n",
            "Epoch 78/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0055 - acc: 0.9969\n",
            "Epoch 79/100\n",
            "8000/8000 [==============================] - 5s 659us/step - loss: 0.0045 - acc: 0.9975\n",
            "Epoch 80/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0045 - acc: 0.9974\n",
            "Epoch 81/100\n",
            "8000/8000 [==============================] - 5s 661us/step - loss: 0.0037 - acc: 0.9975\n",
            "Epoch 82/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0042 - acc: 0.9972\n",
            "Epoch 83/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0149 - acc: 0.9941\n",
            "Epoch 84/100\n",
            "8000/8000 [==============================] - 5s 657us/step - loss: 0.0087 - acc: 0.9965\n",
            "Epoch 85/100\n",
            "8000/8000 [==============================] - 5s 654us/step - loss: 0.0070 - acc: 0.9970\n",
            "Epoch 86/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0051 - acc: 0.9969\n",
            "Epoch 87/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0037 - acc: 0.9974\n",
            "Epoch 88/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0039 - acc: 0.9976\n",
            "Epoch 89/100\n",
            "8000/8000 [==============================] - 5s 659us/step - loss: 0.0084 - acc: 0.9964\n",
            "Epoch 90/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0107 - acc: 0.9964\n",
            "Epoch 91/100\n",
            "8000/8000 [==============================] - 5s 656us/step - loss: 0.0043 - acc: 0.9977\n",
            "Epoch 92/100\n",
            "8000/8000 [==============================] - 5s 663us/step - loss: 0.0039 - acc: 0.9975\n",
            "Epoch 93/100\n",
            "8000/8000 [==============================] - 5s 654us/step - loss: 0.0044 - acc: 0.9977\n",
            "Epoch 94/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0037 - acc: 0.9977\n",
            "Epoch 95/100\n",
            "8000/8000 [==============================] - 5s 659us/step - loss: 0.0036 - acc: 0.9975\n",
            "Epoch 96/100\n",
            "8000/8000 [==============================] - 5s 655us/step - loss: 0.0036 - acc: 0.9976\n",
            "Epoch 97/100\n",
            "8000/8000 [==============================] - 5s 658us/step - loss: 0.0115 - acc: 0.9957\n",
            "Epoch 98/100\n",
            "8000/8000 [==============================] - 5s 652us/step - loss: 0.0092 - acc: 0.9962\n",
            "Epoch 99/100\n",
            "8000/8000 [==============================] - 5s 651us/step - loss: 0.0049 - acc: 0.9974\n",
            "Epoch 100/100\n",
            "8000/8000 [==============================] - 5s 653us/step - loss: 0.0036 - acc: 0.9974\n",
            "accuracy is 0.909\n",
            "roc-auc is 0.910\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}