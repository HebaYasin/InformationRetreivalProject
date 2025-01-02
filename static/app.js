async function searchArticles() {
    const query = document.getElementById("searchBox").value;
    if (query.trim() === '') {
        alert("Please enter a query!");
        return;
    }

    document.getElementById("results").innerHTML = '<div class="loading">🔍 Searching...</div>';

    const response = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    });

    const results = await response.json();
    const resultDiv = document.getElementById("results");
    resultDiv.innerHTML = '';

    results.forEach(article => {
        const link = document.createElement('a');
        link.href = article.link;
        link.textContent = article.title;
        link.target = '_blank';
        link.classList.add('result-item');
        resultDiv.appendChild(link);
    });
}