#![allow(dead_code)]

pub fn minimal_elements<T, R>(items: &[T], rel: R) -> Vec<&T>
where
    R: Fn(&T, &T) -> bool,
{
    let mut mins = Vec::new();
    'outer: for candidate in items {
        for opponent in items {
            if std::ptr::eq(candidate, opponent) {
                continue;
            }
            if rel(opponent, candidate) && !rel(candidate, opponent) {
                continue 'outer;
            }
        }
        mins.push(candidate);
    }
    mins
}

pub fn max_segment_sum(nums: &[i64]) -> i64 {
    let mut best = i64::MIN;
    let mut current = 0;
    for &value in nums {
        current = (current + value).max(value);
        best = best.max(current);
    }
    best
}

#[derive(Clone, Debug)]
pub struct EmployeeNode {
    pub name: String,
    pub rating: i32,
    pub reports: Vec<EmployeeNode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartyPlan {
    pub guests: Vec<String>,
    pub score: i32,
}

impl PartyPlan {
    fn new() -> Self {
        Self {
            guests: Vec::new(),
            score: 0,
        }
    }

    fn with_guest(mut self, guest: String, rating: i32) -> Self {
        self.score += rating;
        self.guests.push(guest);
        self
    }

    fn merge(mut self, other: PartyPlan) -> Self {
        self.score += other.score;
        self.guests.extend(other.guests);
        self
    }
}

pub fn optimal_party(root: &EmployeeNode) -> PartyPlan {
    fn solve(node: &EmployeeNode) -> (PartyPlan, PartyPlan) {
        let mut include_plan = PartyPlan::new().with_guest(node.name.clone(), node.rating);
        let mut exclude_plan = PartyPlan::new();

        for report in &node.reports {
            let (report_include, report_exclude) = solve(report);
            include_plan = include_plan.merge(report_exclude.clone());
            if report_include.score > report_exclude.score {
                exclude_plan = exclude_plan.merge(report_include);
            } else {
                exclude_plan = exclude_plan.merge(report_exclude);
            }
        }

        (include_plan, exclude_plan)
    }

    let (include, exclude) = solve(root);
    if include.score >= exclude.score {
        include
    } else {
        exclude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_elements_identifies_all_minima() {
        let values = [1, 2, 3];
        let mins = minimal_elements(&values, |a, b| a <= b);
        assert_eq!(mins, vec![&1]);
    }

    #[test]
    fn max_segment_sum_matches_reference() {
        let data = [3, -2, 5, -1];
        assert_eq!(max_segment_sum(&data), 6);
    }

    #[test]
    fn optimal_party_selects_best_attendees() {
        let hierarchy = EmployeeNode {
            name: "Root".into(),
            rating: 5,
            reports: vec![
                EmployeeNode {
                    name: "A".into(),
                    rating: 6,
                    reports: vec![],
                },
                EmployeeNode {
                    name: "B".into(),
                    rating: 4,
                    reports: vec![EmployeeNode {
                        name: "C".into(),
                        rating: 10,
                        reports: vec![],
                    }],
                },
            ],
        };
        let plan = optimal_party(&hierarchy);
        assert!(plan.score >= 16);
        assert!(!plan.guests.contains(&"B".into()) || !plan.guests.contains(&"C".into()));
    }
}
