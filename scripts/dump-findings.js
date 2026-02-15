const db = require("better-sqlite3")("/home/snoozyy/ruvnet-research/db/research.db");

const findings = db.prepare(`
  SELECT f.id, f.severity, f.category, f.description, f.followed_up,
    fi.relative_path, fi.loc, p.name as pkg,
    GROUP_CONCAT(DISTINCT d.name) as domains
  FROM findings f
  JOIN files fi ON f.file_id = fi.id
  JOIN packages p ON fi.package_id = p.id
  LEFT JOIN file_domains fd ON fi.id = fd.file_id
  LEFT JOIN domains d ON fd.domain_id = d.id
  GROUP BY f.id
  ORDER BY
    CASE f.severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END,
    fi.relative_path
`).all();

console.log(JSON.stringify(findings, null, 2));
console.log("\nTotal:", findings.length);
console.log("CRITICAL:", findings.filter(f => f.severity === "CRITICAL").length);
console.log("HIGH:", findings.filter(f => f.severity === "HIGH").length);
db.close();
