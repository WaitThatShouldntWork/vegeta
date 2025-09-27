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
SET doc.title = 'Skyfall - Wikipedia', doc.doc_url = 'https://en.wikipedia.org/wiki/Skyfall'
MERGE (sec:Section {doc_url: 'https://en.wikipedia.org/wiki/Skyfall', order: 1})
MERGE (par:Paragraph {doc_url: 'https://en.wikipedia.org/wiki/Skyfall', order: 1})
MERGE (sen:Sentence {doc_url: 'https://en.wikipedia.org/wiki/Skyfall', order: 1, text:'Skyfall won the BAFTA for Outstanding British Film.'})
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

// --- Checklist: VerifyMusicRights (High-risk, Procedure-driven) ---
MERGE (cl2:Checklist {
  name:'VerifyMusicRights',
  description:'Verify music rights compliance through 5-step procedural checklist'
})
MERGE (ss4:SlotSpec {
  checklist_name:'VerifyMusicRights',
  name:'film',
  expect_labels:['Film'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss5:SlotSpec {
  checklist_name:'VerifyMusicRights',
  name:'music_track',
  expect_labels:['MusicTrack'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss6:SlotSpec {
  checklist_name:'VerifyMusicRights',
  name:'composer',
  expect_labels:['Person'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss7:SlotSpec {
  checklist_name:'VerifyMusicRights',
  name:'sync_rights',
  expect_labels:['Document'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (ss8:SlotSpec {
  checklist_name:'VerifyMusicRights',
  name:'territory_clearance',
  expect_labels:['Territory'],
  rel:'INSTANCE_OF',
  required:true,
  cardinality:'ONE'
})
MERGE (cl2)-[:REQUIRES]->(ss4)
MERGE (cl2)-[:REQUIRES]->(ss5)
MERGE (cl2)-[:REQUIRES]->(ss6)
MERGE (cl2)-[:REQUIRES]->(ss7)
MERGE (cl2)-[:REQUIRES]->(ss8);

// --- Additional People (Actors/Directors) ---
MATCH (tPerson:Type {name:'Person'})
WITH tPerson,
[
  {id:'person:sean_connery',         name:'Sean Connery',           aliases:['Sir Sean Connery']},
  {id:'person:george_lazenby',       name:'George Lazenby',         aliases:[]},
  {id:'person:roger_moore',          name:'Roger Moore',            aliases:['Sir Roger Moore']},
  {id:'person:timothy_dalton',       name:'Timothy Dalton',         aliases:[]},
  {id:'person:keira_knightley',      name:'Keira Knightley',        aliases:[]},
  {id:'person:christoph_waltz',      name:'Christoph Waltz',        aliases:[]},
  {id:'person:ralph_fiennes',        name:'Ralph Fiennes',          aliases:[]},
  {id:'person:naomie_harris',        name:'Naomie Harris',          aliases:[]},
  {id:'person:ben_whishaw',          name:'Ben Whishaw',            aliases:[]},
  {id:'person:jeff_bridges',         name:'Jeff Bridges',           aliases:[]},
  {id:'person:olivia_wilde',         name:'Olivia Wilde',           aliases:[]},
  {id:'person:michael_fassbender',   name:'Michael Fassbender',     aliases:[]},
  {id:'person:alec_baldwin',         name:'Alec Baldwin',           aliases:[]},
  {id:'person:bryan_cranston',       name:'Bryan Cranston',         aliases:[]},
  {id:'person:scarlett_johansson',   name:'Scarlett Johansson',     aliases:[]},
  {id:'person:adele',               name:'Adele',                   aliases:['Adele Adkins']},
  {id:'person:eric_serra',          name:'Éric Serra',              aliases:[]},
  {id:'person:don_davis',           name:'Don Davis',               aliases:[]},
  {id:'person:hans_zimmer',         name:'Hans Zimmer',             aliases:[]}
] AS more_people
UNWIND more_people AS p
MERGE (person:Entity:Person {id: p.id})
SET person.name = p.name,
    person.aliases = p.aliases
MERGE (person)-[:INSTANCE_OF]->(tPerson);

// --- More Cast Relationships ---
WITH [
  // Bond actors
  {person:'person:sean_connery',       film:'film:dr_no'},
  {person:'person:sean_connery',       film:'film:goldeneye'},
  {person:'person:george_lazenby',     film:'film:the_world_is_not_enough'},
  {person:'person:roger_moore',        film:'film:the_world_is_not_enough'},
  {person:'person:timothy_dalton',     film:'film:casino_royale'},
  {person:'person:keira_knightley',    film:'film:casino_royale'},
  {person:'person:christoph_waltz',    film:'film:casino_royale'},
  {person:'person:ralph_fiennes',      film:'film:skyfall'},
  {person:'person:naomie_harris',      film:'film:skyfall'},
  {person:'person:ben_whishaw',        film:'film:skyfall'},

  // Other films
  {person:'person:jeff_bridges',       film:'film:true_lies'},
  {person:'person:olivia_wilde',       film:'film:tron_legacy'},
  {person:'person:michael_fassbender', film:'film:tron_legacy'},
  {person:'person:alec_baldwin',       film:'film:mission_impossible_1'},
  {person:'person:bryan_cranston',     film:'film:argo'},
  {person:'person:scarlett_johansson', film:'film:the_matrix'}
] AS more_cast
UNWIND more_cast AS r
MATCH (p:Entity:Person {id: r.person})
MATCH (f:Entity:Film   {id: r.film})
MERGE (p)-[:ACTED_IN]->(f);

// --- More Awards and Facts ---
MATCH (tAward:Type {name:'Award'})
MERGE (oscar:Entity:Award {id:'award:academy_award', name:'Academy Award'})
MERGE (oscar)-[:INSTANCE_OF]->(tAward)
MERGE (golden_globe:Entity:Award {id:'award:golden_globe', name:'Golden Globe'})
MERGE (golden_globe)-[:INSTANCE_OF]->(tAward)
MERGE (sag:Entity:Award {id:'award:sag', name:'Screen Actors Guild Award'})
MERGE (sag)-[:INSTANCE_OF]->(tAward);

// --- Additional Award Facts ---
MATCH (skyfall:Entity:Film {id:'film:skyfall'})
MATCH (argo:Entity:Film {id:'film:argo'})
MATCH (artist:Entity:Film {id:'film:the_artist'})
MATCH (spotlight:Entity:Film {id:'film:spotlight'})
MATCH (green_book:Entity:Film {id:'film:green_book'})

// Create more award facts
MERGE (fact1:Fact {kind:'WON_AWARD'})
SET fact1.confidence = 1.0
MERGE (fact1)-[:SUBJECT]->(argo)
MERGE (fact1)-[:PREDICATE]->(:RelationType {name:'WON_AWARD'})
MERGE (fact1)-[:OBJECT]->(oscar)

MERGE (fact2:Fact {kind:'WON_AWARD'})
SET fact2.confidence = 1.0
MERGE (fact2)-[:SUBJECT]->(spotlight)
MERGE (fact2)-[:PREDICATE]->(:RelationType {name:'WON_AWARD'})
MERGE (fact2)-[:OBJECT]->(oscar)

MERGE (fact3:Fact {kind:'WON_AWARD'})
SET fact3.confidence = 1.0
MERGE (fact3)-[:SUBJECT]->(green_book)
MERGE (fact3)-[:PREDICATE]->(:RelationType {name:'WON_AWARD'})
MERGE (fact3)-[:OBJECT]->(oscar);

// --- Rating Facts ---
MERGE (rating_type:Type {name:'Rating'})
MERGE (r_rating:Entity:Rating {id:'rating:r', name:'R'})
MERGE (r_rating)-[:INSTANCE_OF]->(rating_type)

WITH [
  {film:'film:skyfall', rating:7.8},
  {film:'film:argo', rating:7.7},
  {film:'film:casino_royale', rating:8.0},
  {film:'film:the_matrix', rating:8.7},
  {film:'film:heat', rating:8.2},
  {film:'film:inception', rating:8.8}
] AS film_ratings
UNWIND film_ratings AS fr
MATCH (f:Entity:Film {id: fr.film})
MERGE (fact_rating:Fact {kind:'HAS_RATING'})
SET fact_rating.confidence = 1.0, fact_rating.value = fr.rating
MERGE (fact_rating)-[:SUBJECT]->(f)
MERGE (fact_rating)-[:PREDICATE]->(:RelationType {name:'HAS_RATING'})
MERGE (fact_rating)-[:OBJECT]->(r:Entity:Rating {id:'rating:r'});

// --- Genre Facts ---
MERGE (genre_type:Type {name:'Genre'})
WITH [
  {film:'film:skyfall', genres:['Action', 'Spy', 'Thriller']},
  {film:'film:argo', genres:['Drama', 'Thriller', 'History']},
  {film:'film:casino_royale', genres:['Action', 'Spy', 'Adventure']},
  {film:'film:the_matrix', genres:['Sci-Fi', 'Action', 'Adventure']},
  {film:'film:heat', genres:['Crime', 'Thriller', 'Drama']}
] AS film_genres
UNWIND film_genres AS fg
MATCH (f:Entity:Film {id: fg.film})
UNWIND fg.genres AS genre_name
MERGE (g:Entity:Genre {id: 'genre:' + toLower(genre_name), name: genre_name})
MERGE (g)-[:INSTANCE_OF]->(genre_type)
MERGE (fact_genre:Fact {kind:'HAS_GENRE'})
SET fact_genre.confidence = 1.0
MERGE (fact_genre)-[:SUBJECT]->(f)
MERGE (fact_genre)-[:PREDICATE]->(:RelationType {name:'HAS_GENRE'})
MERGE (fact_genre)-[:OBJECT]->(g);

// Award checklist removed to focus on core procedure-driven system

// Rating checklist removed to focus on core procedure-driven system

// --- Music Rights Data Structures ---

// Music Types
MERGE (tMusicTrack:Type {name:'MusicTrack'})
MERGE (tTerritory:Type {name:'Territory'})

// Sample Music Tracks with Composers
WITH [
  {id:'music:skyfall_theme', name:'Skyfall (Adele)', composer:'person:adele'},
  {id:'music:goldeneye_theme', name:'GoldenEye Theme', composer:'person:eric_serra'},
  {id:'music:matrix_theme', name:'Matrix Theme', composer:'person:don_davis'},
  {id:'music:heat_score', name:'Heat Score', composer:'person:hans_zimmer'}
] AS tracks
UNWIND tracks AS track
MERGE (mt:Entity:MusicTrack {id: track.id})
SET mt.name = track.name
MERGE (mt)-[:INSTANCE_OF]->(tMusicTrack)
MERGE (composer:Entity:Person {id: track.composer})
MERGE (mt)-[:COMPOSED_BY]->(composer)

// Territories
WITH [
  {id:'territory:worldwide', name:'Worldwide'},
  {id:'territory:usa', name:'United States'},
  {id:'territory:eu', name:'European Union'},
  {id:'territory:uk', name:'United Kingdom'},
  {id:'territory:canada', name:'Canada'}
] AS territories
UNWIND territories AS territory
MERGE (t:Entity:Territory {id: territory.id})
SET t.name = territory.name
MERGE (t)-[:INSTANCE_OF]->(tTerritory)

// Simple Music Rights Data (Procedural Focus)
// Skyfall has ONLY the film SlotValue - system must ask for the other 4 required SlotSpecs
WITH ['film:skyfall'] AS film_ids
UNWIND film_ids AS fid
MATCH (film:Entity:Film {id: fid})

// Create ONLY the film SlotValue (system knows about the film)
MERGE (sv_film:SlotValue {slot:'film', value:fid})
MERGE (film)-[:HAS_SLOT]->(sv_film)

// NOTE: System will need to ASK for:
// - music_track (what music track is used?)
// - composer (who composed it?)
// - sync_rights (do we have sync rights documentation?)
// - territory_clearance (what territories are cleared?)

// Create some sample documents for when user provides answers
MERGE (sync_doc_sample:Document {
  source_url:'https://contracts.example.com/sync-rights-template.pdf',
  title:'Sync Rights Template Document'
})
MERGE (territory_worldwide:Entity:Territory {id:'territory:worldwide'})
SET territory_worldwide.name = 'Worldwide'

// --- Sample counts ---
RETURN
  'films'      AS label,  count { ( :Entity:Film ) } AS n_films,
  'people'     AS label2, count { ( :Entity:Person ) } AS n_people,
  'music_tracks' AS label3, count { ( :Entity:MusicTrack ) } AS n_tracks,
  'territories' AS label4, count { ( :Entity:Territory ) } AS n_territories,
  'slots'      AS label5, count { ( :SlotValue ) } AS n_slots,
  'facts'      AS label6, count { ( :Fact ) } AS n_facts,
  'awards'     AS label7, count { ( :Entity:Award ) } AS n_awards,
  'documents'  AS label8, count { ( :Document ) } AS n_documents,
  'checklists' AS label9, count { ( :Checklist ) } AS n_checklists;