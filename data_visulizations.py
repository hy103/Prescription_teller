import matplotlib.pyplot as plt
import seaborn as sns

reason_counts = df.groupby("Reason").size()

# Plotting
plt.figure(figsize=(20, 6))
sns.barplot(x=reason_counts.index, y=reason_counts.values, palette="viridis")
plt.xlabel("Reason")
plt.ylabel("Count")
plt.title("Count of Each Category in 'Reason'")
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.show()


disease_counts = df["Reason"].value_counts()

explode = [0.05] * len(disease_counts) 

plt.figure(figsize=(10, 10))
plt.pie(disease_counts, labels=disease_counts.index, 
        autopct='%1.1f%%', startangle=140, counterclock=False,
        explode=explode, pctdistance=0.85, labeldistance=1.1)


centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')  
plt.title("Category Distribution")
plt.show()


# bar plot
plt.figure(figsize=(12, 12))
ax = disease_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Counts of Categories')
plt.xlabel('Category')
plt.ylabel('Count')

for i in ax.containers:
    ax.bar_label(i, label_type='edge', padding=3, rotation=45)

# Display the plot
plt.show()
plt.show()


all_text = ' '.join(new_df["cleaned_Description"])
words = all_text.split()
word_series = pd.Series(words)
word_counts = word_series.value_counts()
top_10_words = word_counts.head(20)

# Plot the top 10 words and their counts
plt.figure(figsize=(10, 6))
top_10_words.plot(kind='bar', color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Top 20 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Counts')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()