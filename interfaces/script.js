function updateDashboard() {
  fetch('/data')
      .then(response => {
          if (!response.ok) throw new Error('Unauthorized');
          return response.json();
      })
      .then(data => {
          document.getElementById('stage').textContent = data.stage || 'N/A';
          document.getElementById('understanding-bar').style.width = `${(data.understanding * 100).toFixed(2)}%`;
          document.getElementById('confidence-bar').style.width = `${(data.confidence * 100).toFixed(2)}%`;
          document.getElementById('compliance').textContent = data.core_values_compliance || 'N/A';

          const personalityDiv = document.getElementById('personality');
          personalityDiv.innerHTML = '';
          for (const [trait, value] of Object.entries(data.personality)) {
              const div = document.createElement('div');
              div.textContent = `${trait}: ${value.toFixed(3)}`;
              personalityDiv.appendChild(div);
          }

          const philosophyUl = document.getElementById('philosophy');
          philosophyUl.innerHTML = '';
          data.philosophy.forEach(insight => {
              const li = document.createElement('li');
              li.textContent = insight;
              philosophyUl.appendChild(li);
          });

          const eventsSpan = document.getElementById('evolution-events');
          eventsSpan.textContent = data.evolution_events
              .map(event => `${event.type} (Node ${event.node_id}, Iter ${event.iteration})`)
              .join(', ');

          // Refresh neural visualization
          const neuralViz = document.getElementById('neural-viz');
          neuralViz.src = `/visualize?t=${new Date().getTime()}`;
      })
      .catch(error => console.error('Error fetching data:', error));
}

// Update every 5 seconds
setInterval(updateDashboard, 5000);
updateDashboard();