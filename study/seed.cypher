cypher_seed = """
// --- Demo cleanup (scoped) ---
MATCH (n:Demo) DETACH DELETE n;

// --- Constraints (id/name uniqueness) ---
CREATE CONSTRAINT IF NOT EXISTS
FOR (f:Film:Demo) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (p:Person:Demo) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (g:Genre:Demo) REQUIRE g.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (y:Year:Demo) REQUIRE y.value IS UNIQUE;

// --- Films ---
WITH [
  {id:'film:goldeneye',              title:'GoldenEye',                       year:1995, genres:['Action','Spy'],        plot:'Bond faces Janus; notable tank chase in St. Petersburg.'},
  {id:'film:skyfall',                title:'Skyfall',                         year:2012, genres:['Action','Spy'],        plot:'Bond protects M from a former MI6 agent seeking revenge.'},
  {id:'film:the_matrix',             title:'The Matrix',                      year:1999, genres:['Sci-Fi','Action'],     plot:'A hacker learns reality is a simulation; chooses the red pill.'},
  {id:'film:mission_impossible_1',   title:'Mission: Impossible',             year:1996, genres:['Action','Spy'],        plot:'Ethan Hunt goes rogue to clear his name after a botched op.'},
  {id:'film:heat',                   title:'Heat',                            year:1995, genres:['Crime','Thriller'],    plot:'A meticulous thief and a relentless detective collide in LA.'},
  {id:'film:bourne_identity',        title:'The Bourne Identity',             year:2002, genres:['Action','Thriller'],   plot:'An amnesiac operative hunts his past while being hunted.'},
  {id:'film:casino_royale',          title:'Casino Royale',                   year:2006, genres:['Action','Spy'],        plot:'Bond’s first 00 mission targets a financier at high-stakes poker.'},
  {id:'film:red_october',            title:'The Hunt for Red October',        year:1990, genres:['Thriller','Spy'],      plot:'A Soviet sub captain may be defecting; CIA analyst investigates.'},
  {id:'film:tinker_tailor',          title:'Tinker Tailor Soldier Spy',       year:2011, genres:['Drama','Spy'],         plot:'Smiley hunts a Soviet mole inside British intelligence.'},
  {id:'film:die_another_day',        title:'Die Another Day',                 year:2002, genres:['Action','Spy'],        plot:'Bond uncovers a plot involving a space weapon and diamonds.'},
  {id:'film:ronin',                  title:'Ronin',                           year:1998, genres:['Action','Thriller'],   plot:'Ex-operatives chase a mysterious briefcase through Europe.'},
  {id:'film:true_lies',              title:'True Lies',                       year:1994, genres:['Action','Comedy'],     plot:'A secret agent’s double life collides with a terrorist plot.'}
] AS films
UNWIND films AS f
MERGE (film:Film:Demo {id: f.id})
SET film.title = f.title,
    film.year  = f.year,
    film.genres = f.genres,
    film.plot  = f.plot
WITH f, film
UNWIND f.genres AS gname
MERGE (g:Genre:Demo {name: gname})
MERGE (film)-[:IN_GENRE]->(g)
WITH f, film
MERGE (y:Year:Demo {value: f.year})
MERGE (film)-[:RELEASE_YEAR]->(y);

// --- People ---
WITH [
  {id:'person:pierce_brosnan', name:'Pierce Brosnan'},
  {id:'person:daniel_craig',   name:'Daniel Craig'},
  {id:'person:keanu_reeves',   name:'Keanu Reeves'},
  {id:'person:tom_cruise',     name:'Tom Cruise'},
  {id:'person:al_pacino',      name:'Al Pacino'},
  {id:'person:robert_de_niro', name:'Robert De Niro'}
] AS people
UNWIND people AS p
MERGE (person:Person:Demo {id: p.id})
SET person.name = p.name;

// --- Cast (subset, enough for early tests) ---
WITH [
  {person:'person:pierce_brosnan', film:'film:goldeneye'},
  {person:'person:pierce_brosnan', film:'film:die_another_day'},
  {person:'person:daniel_craig',   film:'film:skyfall'},
  {person:'person:daniel_craig',   film:'film:casino_royale'},
  {person:'person:keanu_reeves',   film:'film:the_matrix'},
  {person:'person:tom_cruise',     film:'film:mission_impossible_1'},
  {person:'person:al_pacino',      film:'film:heat'},
  {person:'person:robert_de_niro', film:'film:heat'},
  {person:'person:robert_de_niro', film:'film:ronin'}
] AS roles
UNWIND roles AS r
MATCH (p:Person:Demo {id: r.person})
MATCH (f:Film:Demo   {id: r.film})
MERGE (p)-[:ACTED_IN]->(f);

// --- Minimal indexes done. Sample counts for sanity (optional) ---
RETURN
  'films'  AS label, count { ( :Film:Demo ) } AS n_films,
  'people' AS label2, count { ( :Person:Demo ) } AS n_people,
  'genres' AS label3, count { ( :Genre:Demo ) } AS n_genres;
"""