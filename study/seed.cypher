MATCH (n) DETACH DELETE n;

// --- Constraints ---
CREATE CONSTRAINT IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (t:Type) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (r:RelationType) REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (sv:SlotValue) REQUIRE (sv.slot, sv.value) IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_url IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (s:Section) REQUIRE (s.doc_url, s.order) IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (p:Paragraph) REQUIRE (p.doc_url, p.order) IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS
FOR (snt:Sentence) REQUIRE (snt.doc_url, snt.order) IS UNIQUE;

// Optional fulltext for lookup (may require admin role)
// CALL db.index.fulltext.createNodeIndex('entity_name_aliases', ['Entity'], ['name','aliases']);

// --- Types registry ---
MERGE (tFilm:Type {name:'Film'});
MERGE (tPerson:Type {name:'Person'});
MERGE (tYear:Type {name:'Year'});
MERGE (tAward:Type {name:'Award'});

// --- Relation registry ---
MERGE (:RelationType {name:'ACTED_IN'});
MERGE (:RelationType {name:'RELEASE_YEAR'});
MERGE (:RelationType {name:'HAS_SLOT'});
MERGE (:RelationType {name:'WON_AWARD'});

// --- Films (as Entities) ---
MATCH (tFilm:Type {name:'Film'})
MATCH (tYear:Type {name:'Year'})
WITH tFilm, tYear,
[
  {id:'film:goldeneye',              name:'GoldenEye',                       year:1995, genres:['Action','Spy'],        aliases:['Golden Eye'], plot:'Bond faces Janus; notable tank chase in St. Petersburg.'},
  {id:'film:skyfall',                name:'Skyfall',                         year:2012, genres:['Action','Spy'],        aliases:[],             plot:'Bond protects M from a former MI6 agent seeking revenge.'},
  {id:'film:the_matrix',             name:'The Matrix',                      year:1999, genres:['Sci-Fi','Action'],     aliases:['Matrix'],     plot:'A hacker learns reality is a simulation; chooses the red pill.'},
  {id:'film:mission_impossible_1',   name:'Mission: Impossible',             year:1996, genres:['Action','Spy'],        aliases:['MI1'],        plot:'Ethan Hunt goes rogue to clear his name after a botched op.'},
  {id:'film:heat',                   name:'Heat',                            year:1995, genres:['Crime','Thriller'],    aliases:[],             plot:'A meticulous thief and a relentless detective collide in LA.'},
  {id:'film:bourne_identity',        name:'The Bourne Identity',             year:2002, genres:['Action','Thriller'],   aliases:[],             plot:'An amnesiac operative hunts his past while being hunted.'},
  {id:'film:casino_royale',          name:'Casino Royale',                   year:2006, genres:['Action','Spy'],        aliases:[],             plot:'Bond’s first 00 mission targets a financier at high-stakes poker.'},
  {id:'film:red_october',            name:'The Hunt for Red October',        year:1990, genres:['Thriller','Spy'],      aliases:['Red October'], plot:'A Soviet sub captain may be defecting; CIA analyst investigates.'},
  {id:'film:tinker_tailor',          name:'Tinker Tailor Soldier Spy',       year:2011, genres:['Drama','Spy'],         aliases:['Tinker Tailor'], plot:'Smiley hunts a Soviet mole inside British intelligence.'},
  {id:'film:die_another_day',        name:'Die Another Day',                 year:2002, genres:['Action','Spy'],        aliases:[],             plot:'Bond uncovers a plot involving a space weapon and diamonds.'},
  {id:'film:ronin',                  name:'Ronin',                           year:1998, genres:['Action','Thriller'],   aliases:[],             plot:'Ex-operatives chase a mysterious briefcase through Europe.'},
  {id:'film:true_lies',              name:'True Lies',                       year:1994, genres:['Action','Comedy'],     aliases:[],             plot:'A secret agent’s double life collides with a terrorist plot.'}
] AS films
UNWIND films AS f
MERGE (film:Entity:Film {id: f.id})
SET film.name = f.name,
    film.title = f.name,
    film.plot  = f.plot,
    film.aliases = f.aliases
MERGE (film)-[:INSTANCE_OF]->(tFilm)
MERGE (y:Entity:Year {id: 'year:' + toString(f.year), value: f.year})
MERGE (y)-[:INSTANCE_OF]->(tYear)
MERGE (film)-[:RELEASE_YEAR]->(y)
WITH f, film
UNWIND f.genres AS gname
MERGE (svg:SlotValue {slot:'Genre', value:gname})
MERGE (film)-[:HAS_SLOT]->(svg);

// AwardsSignal
WITH ['film:skyfall','film:casino_royale','film:heat'] AS ids
UNWIND ids AS fid
MATCH (film:Entity:Film {id: fid})
WITH film, CASE film.id WHEN 'film:skyfall' THEN 'High' WHEN 'film:casino_royale' THEN 'Medium' ELSE 'Low' END AS sig
MERGE (svs:SlotValue {slot:'AwardsSignal', value:sig})
MERGE (film)-[:HAS_SLOT]->(svs);

// --- People ---
MATCH (tPerson:Type {name:'Person'})
WITH tPerson,
[
  {id:'person:pierce_brosnan', name:'Pierce Brosnan', aliases:['Pierce Brendan Brosnan']},
  {id:'person:daniel_craig',   name:'Daniel Craig',   aliases:[]},
  {id:'person:keanu_reeves',   name:'Keanu Reeves',   aliases:[]},
  {id:'person:tom_cruise',     name:'Tom Cruise',     aliases:[]},
  {id:'person:al_pacino',      name:'Al Pacino',      aliases:[]},
  {id:'person:robert_de_niro', name:'Robert De Niro', aliases:['Bob De Niro']}
] AS people
UNWIND people AS p
MERGE (person:Entity:Person {id: p.id})
SET person.name = p.name,
    person.aliases = p.aliases
MERGE (person)-[:INSTANCE_OF]->(tPerson);

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
MATCH (p:Entity:Person {id: r.person})
MATCH (f:Entity:Film   {id: r.film})
MERGE (p)-[:ACTED_IN]->(f);

// --- Provenance + one reified fact (Skyfall won BAFTA) ---
MATCH (tAward:Type {name:'Award'})
MERGE (aw:Entity:Award {id:'award:bafta', name:'BAFTA Award'})
MERGE (aw)-[:INSTANCE_OF]->(tAward)
MERGE (doc:Document {source_url:'https://en.wikipedia.org/wiki/Skyfall'})
SET doc.title = 'Skyfall - Wikipedia', doc.doc_url = doc.source_url
MERGE (sec:Section {doc_url: doc.source_url, order: 1})
MERGE (par:Paragraph {doc_url: doc.source_url, order: 1})
MERGE (sen:Sentence {doc_url: doc.source_url, order: 1, text:'Skyfall won the BAFTA for Outstanding British Film.'})
MERGE (doc)-[:HAS_SECTION]->(sec)
MERGE (sec)-[:HAS_PARAGRAPH]->(par)
MERGE (par)-[:HAS_SENTENCE]->(sen);

MATCH (sky:Entity:Film {id:'film:skyfall'})
MATCH (aw:Entity:Award {id:'award:bafta'})
MATCH (doc:Document {source_url:'https://en.wikipedia.org/wiki/Skyfall'})
MERGE (rt:RelationType {name:'WON_AWARD'})
MERGE (fact:Fact {kind:'WON_AWARD'})
MERGE (fact)-[:SUBJECT]->(sky)
MERGE (fact)-[:PREDICATE]->(rt)
MERGE (fact)-[:OBJECT]->(aw)
MERGE (fact)-[:HAS_SOURCE {support:1.0}]->(doc);

// Optional mentions
MATCH (sen:Sentence {doc_url:'https://en.wikipedia.org/wiki/Skyfall', order: 1})
MATCH (sky:Entity:Film {id:'film:skyfall'})
MATCH (aw:Entity:Award {id:'award:bafta'})
MERGE (sen)-[:MENTIONS {confidence:0.9, via:'seed'}]->(sky)
MERGE (sen)-[:MENTIONS {confidence:0.9, via:'seed'}]->(aw);

// --- Checklist: IdentifyFilm ---
MERGE (cl:Checklist {name:'IdentifyFilm', description:'Identify a specific film from clues'})
MERGE (ss1:SlotSpec {checklist_name:'IdentifyFilm', name:'film', expect_labels:['Film'], rel:'INSTANCE_OF', required:true, cardinality:'ONE'})
MERGE (ss2:SlotSpec {checklist_name:'IdentifyFilm', name:'year', expect_labels:['Year'], required:false, cardinality:'ONE'})
MERGE (ss3:SlotSpec {checklist_name:'IdentifyFilm', name:'actor', expect_labels:['Person'], required:false, cardinality:'MANY'})
MERGE (cl)-[:REQUIRES]->(ss1)
MERGE (cl)-[:REQUIRES]->(ss2)
MERGE (cl)-[:REQUIRES]->(ss3);

// --- Sample counts ---
RETURN
  'films'   AS label, count { ( :Entity:Film ) } AS n_films,
  'people'  AS label2, count { ( :Entity:Person ) } AS n_people,
  'slots'   AS label3, count { ( :SlotValue ) } AS n_slots;